import geopandas as gpd  # type: ignore
import matplotlib.dates as mdates  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import pybaselines as pybs  # type: ignore
import skgstat as skg  # type: ignore
from scipy import integrate, odr  # type: ignore
from scipy.optimize import least_squares  # type: ignore


# functions for SeekOps data
def SeekOps_pre_process(df):
    df = timestamp_from_four_columns(df)
    df.rename(
        columns={
            'Latitude': 'latitude',
            'Longitude': 'longitude',
            'Altitude_m': 'altitude',
            'Methane_ppb': 'ch4',
            'WindSpeed_ms': 'windspeed',
            'WindDirection': 'winddir',
        },
        inplace=True,
    )
    df.ch4 = df.ch4 / 1000
    df = add_utm(df)
    return df


# functions for Scientific Aviation data
def SciAv_pre_process(folder):
    assert (
        len(list(folder.glob('merge.txt'))) == 1
    ), 'more than one merge.txt file found'
    df = pd.read_csv(list(folder.glob('merge.txt'))[0])
    df2 = pd.DataFrame()
    # read all raw files and append to df2
    for file in folder.glob('Z*.txt'):
        df_t = pd.read_csv(file, skiprows=1)
        df_t.columns = [
            'Time Since 1970',
            'wU',
            'wV',
            'wW',
            'a_u',
            'a_v',
            'a_w',
            'Pitch',
            'Roll',
            'Yaw',
            'Latitude',
            'Longitude',
            'Altitude',
            'IsFlying?',
            'BB Time',
            'CH4',
            'C2',
            'CO2',
            'V1',
            'V2',
            'V3',
            'V4',
        ]
        df2 = pd.concat([df2, df_t], axis=0)
    # get altitude, lat and lom from df2 by merging on epoch time
    df2['time'] = pd.to_datetime(df2['Time Since 1970'], unit='s').dt.time.astype(str)
    df2 = df2[['time', 'Altitude', 'Latitude', 'Longitude']]
    df2['time'] = pd.to_datetime(df2['time'])
    df.drop(columns=['Latitude', 'Longitude'], inplace=True)
    df['time'] = pd.to_datetime(df['Time(EPOCH)'], unit='s').dt.time.astype(str)
    df['time'] = pd.to_datetime(df['time'])
    df = pd.merge_asof(
        df, df2, on='time', direction='nearest', tolerance=pd.Timedelta('2s')
    )
    # reindex to UTC(?) from MST
    df.index = df['Time(MST)'].apply(
        lambda x: pd.Timestamp(x, tz='MST').tz_convert('UTC')
    )
    df.index.name = 'Time(UTC)'
    df['epoch'] = df['Time(EPOCH)'].apply(lambda x: pd.Timestamp(x, unit='s', tz='UTC'))
    df['seconds'] = df['epoch'].dt.second + df['epoch'].dt.microsecond / 1e6
    df.index = df.index.floor('1min') + pd.to_timedelta(df['seconds'], unit='s')
    df = df.drop(
        columns=['Time(MST)', 'Time(EPOCH)', 'epoch', 'seconds', 'Time3', 'time']
    )
    df.rename(
        columns={
            'Latitude': 'latitude',
            'Longitude': 'longitude',
            'Wind Speed (m/s)': 'windspeed',
            'Wind Dir (deg)': 'winddir',
            'Methane(ppm)': 'ch4',
            'Ethane': 'c2h6',
            'Altitude': 'altitude',
        },
        inplace=True,
    )
    assert df.index.is_unique, 'duplicate indices found'
    df = add_utm(df)
    return df


def SciAv_process(
    df,
    celsius,
    millibars,
    odr_distance_filter,
    azimuth_filter,
    azimuth_window,
    elevation_filter,
):
    df, startrow, endrow = heuristic_row_filter(
        df,
        azimuth_filter=azimuth_filter,
        azimuth_window=azimuth_window,
        elevation_filter=elevation_filter,
    )
    df, plane_angle = flatten_linear_plane(df, odr_distance_filter)
    df = methane_flux_column(df, celsius, millibars)
    df = x_filter(df, startrow, endrow)
    return df


def SeekOps_process(df, celsius, millibars):  # after baseline correction
    df['circ_deviation'], df['circ_azimuth'], df_radius = circle_deviation(
        df, 'utm_easting', 'utm_northing'
    )
    df = df[
        df['circ_deviation'].between(-df_radius / 10, df_radius / 10)
    ].copy()  # 10% radius tolerance
    df = recentre_azimuth(df, r=df_radius)
    df = methane_flux_column(df, celsius, millibars)
    return df


# pre-processing functions

# open csv file
def open_csv(df):
    df = pd.read_csv(df)
    return df


# make timestamp column from UTCs, Month, Day, Year
def timestamp_from_four_columns(df):
    df['Year'] = df['Year'] + 2000
    df['time'] = pd.to_datetime(df['UTCs'], unit='s')
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df['timestamp'] = pd.to_datetime(
        df['date'].dt.date.astype(str) + ' ' + df['time'].dt.time.astype(str)
    )
    df.index = df['timestamp']
    df.drop(
        ['Year', 'Month', 'Day', 'time', 'date', 'timestamp', 'UTCs'],
        axis=1,
        inplace=True,
    )
    return df


# add UTM from latitudes and longitudes
def add_utm(df):
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['longitude'], df['latitude'], crs='EPSG:4326'),
    )
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)
    df['utm_easting'] = gdf.geometry.x
    df['utm_northing'] = gdf.geometry.y
    return df


# add columns for drone bearings
def heading(df):
    df['hor_distance'] = np.sqrt(
        (df['utm_northing'].diff()) ** 2 + (df['utm_easting'].diff()) ** 2
    )
    df['vert_distance'] = df['altitude'].diff()

    df['elevation_heading'] = np.degrees(
        np.arctan2(df['vert_distance'], df['hor_distance'])
    )
    df['azimuth_heading'] = (
        np.degrees(np.arctan2(df['utm_easting'].diff(), df['utm_northing'].diff()))
        % 360
    )
    return df


# heuristic filtering to produce rows of data

# function to return the start and end of the biggest monotonic series of values from a dictionary
def mCount(dict):
    poscount = 0
    negcount = 0
    max_pos_count = 0
    max_pos_row = 0
    max_neg_count = 0
    max_neg_row = 0
    pos_start = 0
    neg_start = 0
    for i in range(1, len(dict)):
        if dict[i] >= dict[i - 1]:
            poscount += 1
            negcount = 0
        elif dict[i] < dict[i - 1]:
            negcount += 1
            poscount = 0
        if max_pos_count < poscount:
            max_pos_count = poscount
            max_pos_row = i
            pos_start = i - poscount
        elif max_neg_count < negcount:
            max_neg_count = negcount
            max_neg_row = i
            neg_start = i - negcount
    if max_pos_count >= max_neg_count:
        return pos_start, max_pos_row
    elif max_neg_count > max_pos_count:
        return neg_start, max_neg_row


def bimodal_azimuth(df):
    data = df['azimuth_heading'].to_numpy()
    hist, edges = np.histogram(data, bins=50)
    max_freq_idx = np.argsort(hist)[::-1][:2]
    mode1, mode2 = edges[max_freq_idx][0], edges[max_freq_idx][1]

    while np.abs(mode1 - mode2) < 160:
        hist[max_freq_idx[1]] = 0
        max_freq_idx = np.argsort(hist)[::-1][:2]
        mode1, mode2 = edges[max_freq_idx][0], edges[max_freq_idx][1]
    assert 160 < np.abs(mode1 - mode2) < 200
    return mode1, mode2


def heuristic_row_filter(
    df,
    azimuth_filter,
    azimuth_window,
    elevation_filter,
):
    df = heading(df)
    df = df[
        abs(df['elevation_heading']) < elevation_filter
    ]  # degrees, filters ups and downs
    azi1, azi2 = bimodal_azimuth(df)
    print(
        f'Drone appears to be flying mainly on the headings {azi1:.2f} degrees and {azi2:.2f} degrees'
    )
    df = df[
        (
            df['azimuth_heading'].rolling(azimuth_window, center=True).mean()
            < azi1 + azimuth_filter
        )
        & (df['azimuth_heading'] > azi1 - azimuth_filter)
        | (df['azimuth_heading'] < azi2 + azimuth_filter)
        & (df['azimuth_heading'] > azi2 - azimuth_filter)
    ]
    df['row'] = 0  # split into lines by incrementing based on azimuth heading switches
    df.loc[df['azimuth_heading'].diff().abs() > 90, 'row'] = 1
    df['row'] = df['row'].cumsum()
    alt_dict = dict(df.groupby('row')['altitude'].mean())
    startrow, endrow = mCount(alt_dict)
    df = df[
        (df['row'] >= startrow) & (df['row'] <= endrow)
    ]  # filter to the biggest monotonic series of values
    print(
        f'Parsed a flight of {endrow-startrow} rows from {alt_dict[startrow]:.0f}m to {alt_dict[endrow]:.0f}m between the time of {df.index[0]} and {df.index[-1]}'
    )
    return df, startrow, endrow


# linear flight path functions
def linear_reg_equation(coefs: tuple[float, float], x: pd.Series) -> pd.Series:
    return coefs[0] * x + coefs[1]  # y = mx + c


def flight_odr_fit(df: pd.DataFrame):
    fit = odr.odr(
        linear_reg_equation, [1, 0], y=df['utm_northing'], x=df['utm_easting']
    )  # inital guess of m=1, c=0
    # add column of distance from linear fit
    df['distance_from_fit'] = np.sqrt(
        (df['utm_northing'] - linear_reg_equation(fit[0], df['utm_easting'])) ** 2
        + (df['utm_easting'] - df['utm_easting']) ** 2
    )
    return df, fit[0]


def flatten_linear_plane(
    df: pd.DataFrame, distance_filter
) -> tuple[pd.DataFrame, float]:
    df, coefs2D = flight_odr_fit(df)
    df = df[
        df['distance_from_fit'] < distance_filter
    ].copy()  # filter to points near the linear fit
    df, coefs2D = flight_odr_fit(df)  # re-fit to filtered points
    df = df[
        df['distance_from_fit'] < distance_filter
    ].copy()  # filter again to points near the linear fit
    rotation = np.arctan(coefs2D[0])
    df['x'] = (df['utm_easting'] - df['utm_easting'].min()) * np.cos(-rotation) - (
        df['utm_northing'] - df['utm_northing'].min()
    ) * np.sin(-rotation)
    df['y'] = (df['utm_easting'] - df['utm_easting'].min()) * np.sin(-rotation) + (
        df['utm_northing'] - df['utm_northing'].min()
    ) * np.cos(-rotation)
    df['z'] = df.altitude
    plane_angle = (np.pi / 2) - np.arctan(coefs2D[0])
    return df, plane_angle


# function to set the edges of the linear plane based on not the top and bottom rows
def x_filter(df, startrow, endrow, x='x', y='z'):
    bb_df = df[(df['row'] > startrow) & (df['row'] < endrow)]
    df = df[(df[x] >= bb_df[x].min()) & (df[x] <= bb_df[x].max())]
    return df


# circular flight path functions
def circle_deviation(df, x=str, y=str):
    # select middle half of data to avoid edge effects
    x, y = df[x], df[y]

    def midhalf(x):
        return x.iloc[int(len(x) * 1 / 4) : int(len(x) * 3 / 4)]

    x_filter, y_filter = midhalf(x), midhalf(y)

    # fit circle to data
    def func(params):
        xc, yc, r = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    # initial guess for parameters
    x_m = np.mean(x_filter)
    y_m = np.mean(y_filter)
    r_m = np.mean(np.sqrt((x_filter - x_m) ** 2 + (y_filter - y_m) ** 2))

    # fit circle
    params0 = np.array([x_m, y_m, r_m])
    result = least_squares(func, params0)
    xc, yc, r = result.x

    # output deviation from circle for filtering purposes
    deviation = np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    # output azimuth in radians with 0 at north and increasing clockwise thanks to modulos
    azimuth = np.degrees(np.arctan2(x - xc, y - yc) % (2 * np.pi))

    return deviation, azimuth, r


def azimuth_of_max(df, x: str = 'circ_azimuth', y: str = 'ch4_normalised'):
    return df.loc[df[y].idxmax()][x]


def recentre_azimuth(df, r: float, x: str = 'circ_azimuth', y: str = 'ch4_normalised'):
    centre_azimuth = azimuth_of_max(df, x, y)
    df['centred_azimuth'] = df[x] - centre_azimuth
    df.loc[df['centred_azimuth'] > 180, 'centred_azimuth'] -= 360
    df.loc[df['centred_azimuth'] < -180, 'centred_azimuth'] += 360
    df['circumference_distance'] = r * np.radians(df['centred_azimuth'])
    df['circumference_distance'] = (
        df['circumference_distance'] - df['circumference_distance'].min()
    )
    return df


# split data into two dataframes, one for each flight, based on the minimum altitude
def splittime(df):
    minalt = df['altitude'].min()
    splittime = df[df['altitude'] == minalt].index[0]
    return splittime


# plotting functions
def scatter_3d(
    df: pd.DataFrame,
    name: str | None = None,
    color: str = 'ch4',
    x: str = 'utm_easting',
    y: str = 'utm_northing',
    z: str = 'altitude',
):
    fig = px.scatter_3d(
        df, x=x, y=y, z=z, color=color, opacity=0.5, color_continuous_scale='geyser'
    )
    fig.update_traces(marker_size=4)
    fig.update_traces(
        customdata=df.index,
        hovertemplate='<br>'.join(
            [
                'northing: %{x:.2f}',
                'easting: %{y:.2f}',
                'altitude: %{z:.2f}',
                'CH4: %{marker.color:.2f}',
                'Time: %{customdata}',
            ]
        ),
    )
    if name:
        fig.write_html(f'{name}.html')
        fig.update_layout(
            title_text=name,
            title_x=0.5,
            title_font_size=20,
        )
    return fig


def scatter_2d(
    df: pd.DataFrame,
    name: str | None = None,
    x: str = 'centred_azimuth',
    y: str = 'altitude',
    color: str = 'ch4_normalised',
    **kwargs,
):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        template='simple_white',
        color_continuous_scale='geyser',
        opacity=0.8,
        **kwargs,
    )
    fig.update_traces(
        customdata=df.index,
        hovertemplate='<br>'.join(
            [
                'x: %{x:.2f}',
                'altitude: %{y:.2f}',
                'CH4: %{marker.color:.2f}',
                'Time: %{customdata}',
            ]
        ),
    )
    if name:
        fig.write_html(f'{name}.html')
        fig.update_layout(
            title_text=name,
            title_x=0.5,
            title_font_size=20,
        )
    return fig


def gas_time_series_plot(df=pd.DataFrame, name=str, gas=str, color=str, split=None):
    fig = px.scatter(df, x=df.index, y=df[gas], color=df[color])
    fig.update_traces(marker_size=8)
    if split is not None:
        y_min, y_max = df[gas].min(), df[gas].max()
        fig.update_layout(
            shapes=[
                dict(
                    type='line',
                    xref='x',
                    yref='y',
                    x0=split,
                    y0=y_min,
                    x1=split,
                    y1=y_max,
                    line=dict(color='red', width=2),
                )
            ]
        )
    fig.update_traces(opacity=0.5)
    if name:
        fig.write_html(f'{name}.html')
    fig.show()


def wind_time_series(df, name=str, y=str, color=str):
    fig = px.scatter(df, x=df.index, y=y, color=color, opacity=0.5)
    fig.update_traces(marker_size=8)
    fig.write_html(f'products/{name}.html')
    fig.show()


# plotting windrose
def windrose_process(df):
    beaufort = {
        '0': [0, 1],
        '1': [1, 2],
        '2': [2, 4],
        '3': [4, 6],
        '4': [6, 9],
        '5': [9, 11],
        '6': [11, 14],
        '7': [14, 17],
        '8': [17, 21],
        '9': [21, 25],
        '10': [25, 29],
        '11': [29, 33],
        '12': [33, 200],
    }

    cardinals = {
        'N1': [0, 11.25],
        'NNE': [11.25, 33.75],
        'NE': [33.75, 56.25],
        'ENE': [56.25, 78.75],
        'E': [78.75, 101.25],
        'ESE': [101.25, 123.75],
        'SE': [123.75, 146.25],
        'SSE': [146.25, 168.75],
        'S': [168.75, 191.25],
        'SSW': [191.25, 213.75],
        'SW': [213.75, 236.25],
        'WSW': [236.25, 258.75],
        'W': [258.75, 281.25],
        'WNW': [281.25, 303.75],
        'NW': [303.75, 326.25],
        'NNW': [326.25, 348.75],
        'N2': [348.75, 360],
    }
    df['direction_bin'] = pd.cut(
        df['winddir'],
        bins=[lower for lower, upper in cardinals.values()]
        + [list(cardinals.values())[-1][1]],
        labels=[key for key in cardinals.keys()],
        right=False,
    )
    df['direction_bin'] = df['direction_bin'].replace({'N1': 'N', 'N2': 'N'})
    df['beaufort'] = pd.cut(
        df['windspeed'],
        bins=[lower for lower, upper in beaufort.values()]
        + [list(beaufort.values())[-1][1]],
        labels=[key for key in beaufort.keys()],
        right=False,
    )
    df_windrose = (
        df.groupby(['direction_bin', 'beaufort']).size().reset_index(name='count')
    )
    df_windrose['frequency'] = df_windrose['count'] / df_windrose['count'].sum() * 100
    df_windrose['direction_degs'] = df_windrose['direction_bin'].replace(
        {
            'N': 0,
            'NNE': 22.5,
            'NE': 45,
            'ENE': 67.5,
            'E': 90,
            'ESE': 112.5,
            'SE': 135,
            'SSE': 157.5,
            'S': 180,
            'SSW': 202.5,
            'SW': 225,
            'WSW': 247.5,
            'W': 270,
            'WNW': 292.5,
            'NW': 315,
            'NNW': 337.5,
        }
    )
    df_windrose['beaufort'] = df_windrose['beaufort'].astype(int)
    return df_windrose


def windrose_graph(df, name):
    n_colors = 13
    colors = px.colors.sample_colorscale(
        'turbo', [n / (n_colors - 1) for n in range(n_colors)]
    )
    fig = px.bar_polar(
        df,
        r='frequency',
        theta='direction_bin',
        color='beaufort',
        template='presentation',
        labels={
            'frequency': 'Frequency (%)',
            'direction_bin': 'Direction',
            'beaufort': 'Beaufort Scale',
        },
        color_discrete_map=colors,
    )
    fig.update_layout(
        polar=dict(radialaxis={'visible': False, 'showticklabels': False})
    )
    fig.update_layout(
        polar=dict(
            angularaxis={
                'showgrid': False,
            }
        )
    )
    fig.update_layout(polar_bargap=0)
    fig.write_html(f'products/{name}.html')
    fig.show()


def windrose(df, name):
    df_windrose = windrose_process(df)
    windrose_graph(df_windrose, name)


def directional_gas_variogram(
    df: pd.DataFrame, x: str, z: str, gas: str, plot: bool = True, **variogram_settings
):
    v = skg.Variogram(
        df[[x, z]].to_numpy(),
        df[gas].to_numpy(),
        **variogram_settings,
    )
    if plot:
        v.plot()
    return v


def plot_contour_krig(
    df: pd.DataFrame, xx: np.ndarray, yy: np.ndarray, field: np.ndarray, x: str, y: str
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode='markers',
            marker={
                'color': df['ch4_kg_h_m2'],
                'colorscale': 'geyser',
                'cmin': field.min(),
                'cmax': field.max(),
            },
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Contour(
            z=field.T,
            x=xx[:, 0],
            y=yy[0, :],
            contours={
                'start': field.min(),
                'end': field.max(),
                'size': field.max() / 21,
            },
            colorscale='geyser',
            opacity=0.5,
            showlegend=False,
        )
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        title_text='horizontal distance on cylindrical projected flux plane (m)',
        range=[xx.min(), xx.max()],
        ticks='outside',
        tickwidth=1,
        tickcolor='black',
        ticklen=5,
        nticks=20,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        title_text='height above ground level (m)',
        range=[yy.min(), yy.max()],
        ticks='outside',
        tickwidth=1,
        tickcolor='black',
        ticklen=5,
        nticks=10,
    )
    fig.layout.coloraxis.colorbar.title = 'Emissions flux (kg⋅m⁻²⋅h⁻¹)'
    fig.update_layout(template='simple_white')
    fig.show()


def plot_heatmap_krig(xx: np.ndarray, yy: np.ndarray, field: np.ndarray):
    fig = px.imshow(
        field.T, x=xx[:, 0], y=yy[0, :], color_continuous_scale='geyser', origin='lower'
    )
    fig.layout.coloraxis.colorbar.title = 'Emissions flux (kg⋅m⁻²⋅h⁻¹)'
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        title_text='horizontal distance on cylindrical projected flux plane (m)',
        range=[xx.min(), xx.max()],
        ticks='outside',
        tickwidth=1,
        tickcolor='black',
        ticklen=5,
        nticks=20,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor='black',
        title_text='height above ground level (m)',
        range=[yy.min(), yy.max()],
        ticks='outside',
        tickwidth=1,
        tickcolor='black',
        ticklen=5,
        nticks=10,
    )
    fig.show()


# data processing
# baselining
# Cobas, J., et al. A new general-purpose fully automatic baseline-correction procedure for 1D and 2D NMR data. Journal of Magnetic Resonance, 2006, 183(1), 145-151.


def baseline(
    df,
    algorithm: str,
    name=None,
    y: str = 'ch4',
    colors: str = 'altitude',
    plot: bool = False,
    **kwargs,
):
    df = df.copy()
    index = np.arange(df.index.shape[0])
    baseline_fitter = pybs.Baseline(index, check_finite=False)
    fit = getattr(baseline_fitter, algorithm)
    bkg, params = fit(df[y], **kwargs)
    bkg_points = params['mask']
    background = (df[y] - bkg)[bkg_points]
    signal = (df[y] - bkg)[~bkg_points]
    if plot:
        x = df.index
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(x, df[y], label='Raw Data', lw=1.5, alpha=0.5)
        ax1.plot(x, bkg, '--', label='Fitted Baseline')

        ax2 = ax1.twinx()
        ax2.set_xlabel('Time')
        ax2.set_ylabel('CH4 (ppm)')

        nml = ax2.scatter(x, df[y] - bkg, label='Normalised Data', c=colors, s=5)
        plt.colorbar(nml, label='Altitude above ground level (m)', pad=0.1)
        y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
        yminax2 = ax2.get_ylim()[0]
        ax2.set_ylim(yminax2, yminax2 + y_range)
        ax2.scatter(
            x[~bkg_points],
            (df[y] - bkg)[~bkg_points],
            label='Methane signal',
            color='red',
            s=7,
        )
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax2.set_ylabel('CH₄ (ppm)')
        ax1.set_ylabel('CH₄ (ppm)')
        fig.legend(
            loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes
        )
        plt.show()
        if name is not None:
            plt.savefig(f'products/{name}_baseline.png')
        else:
            print('No name given for baseline plot, not saving.')
    print(f'Baseline algorithm: {algorithm}')
    print(
        f'Positive and negative 95% percentile of baseline: {np.percentile(background, 2.5):.0f} ppm, {np.percentile(background, 97.5):.0f} ppm'
    )
    print(f'Mean of baseline: {np.mean(background):.2f} ppm')
    print(
        f'Minimum and maximum of baseline: {np.min(background):.2f} ppm, {np.max(background):.2f} ppm'
    )
    print(f'Signal points: {len(signal)}; background points: {len(background)}')
    df[f'{y}_normalised'] = df[y] - bkg
    return df


gas_variables = {
    'standard_pressure': 1013.25,  # mbar
    'standard_temperature': 273.15,  # K
    'methane_molar_mass': 16.04,  # g⋅mol−1
    'standard_methane_molar_volume': 0.022413,  # m3⋅mol−1
}


# def gas_local_volume(df: pd.DataFrame, local_pressure, local_temperature) -> float:
#     return (  # type: ignore
#         standard_molar_volume
#         * (standard_pressure / local_pressure)
#         * ((local_temperature + standard_temperature) / standard_temperature)
#     )  # m3⋅mol−1


# def ch4_density(df: pd.DataFrame) -> float:
#     return CH4_molar_mass / 1000 / gas_local_volume(df)  # kg⋅m-3


def ch4_density(
    df: pd.DataFrame, local_pressure, local_temperature
) -> float:  # millibars and celsius
    methane_local_volume = (
        gas_variables['standard_methane_molar_volume']
        * (gas_variables['standard_pressure'] / local_pressure)
        * (
            (local_temperature + gas_variables['standard_temperature'])
            / gas_variables['standard_temperature']
        )
    )  # m3⋅mol−1
    return gas_variables['methane_molar_mass'] / 1000 / methane_local_volume  # kg⋅m-3


def methane_flux_column(
    df: pd.DataFrame,
    celsius: float,
    millibars: float,
    gas: str = 'ch4',
    wind: str = 'windspeed',
) -> pd.DataFrame:
    methane_density = ch4_density(
        df, local_pressure=millibars, local_temperature=celsius
    )  # kg/m3
    df['ch4_kg_m3'] = methane_density * (df[f'{gas}_normalised'] * 1e-6)  # kg/m3
    df['ch4_kg_h_m2'] = df['ch4_kg_m3'] * df[wind] * 60 * 60  # kg/h/m2
    return df


def simpsonintegrate(
    array: np.ndarray, x_cell_size: float, y_cell_size: float
) -> float:
    """function to obtain the volume of the krig in kgh⁻¹, i.e. the cut-fill volume (negative volumes from baseline noise are subtracted)."""
    # this integrates along each row of the grid
    vol_rows = integrate.simpson(np.transpose(array))
    vol_grid = integrate.simpson(vol_rows)  # this integrates the rows together
    return vol_grid * x_cell_size * y_cell_size  # type: ignore


def ordinary_kriging(
    df: pd.DataFrame,
    x: str,
    y: str,
    gas: str,
    ordinary_kriging_settings: dict,
    plot_variogram=True,
    plot_contours=True,
    plot_grid=False,
    **variogram_settings,
):
    skg.plotting.backend('plotly')
    ok = skg.OrdinaryKriging(
        variogram=directional_gas_variogram(
            df, x, y, gas, plot=plot_variogram, **variogram_settings
        ),
        coordinates=df[[x, y]].to_numpy(),
        values=df[gas].to_numpy(),
        min_points=ordinary_kriging_settings['min_points'],
        max_points=ordinary_kriging_settings['max_points'],
    )
    x_max = df[x].max()
    x_min = df[x].min()
    y_max = df[y].max()
    y_min = df[y].min()
    x_range, y_range = df[x].max() - df[x].min(), df[y].max() - df[y].min()
    cell_rough_size = np.sqrt(
        (x_range * y_range) / ordinary_kriging_settings['grid_resolution']
    )
    x_nodes, y_nodes = [
        max(int(r / cell_rough_size), ordinary_kriging_settings['min_nodes'])
        for r in [x_range, y_range]
    ]
    x_cell_size = (x_max - x_min) / x_nodes
    y_cell_size = (y_max - y_min) / y_nodes
    xx, yy = np.mgrid[
        x_min : x_max : x_nodes * 1j, y_min : y_max : y_nodes * 1j  # type: ignore
    ]  # type: ignore
    field = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)

    np.nan_to_num(field, copy=False, nan=0)
    volume = simpsonintegrate(field, x_cell_size, y_cell_size)

    fieldpos = np.copy(field)
    fieldpos[fieldpos < 0] = 0
    volumepos = simpsonintegrate(fieldpos, x_cell_size, y_cell_size)

    fieldneg = np.copy(field)
    fieldneg[fieldneg > 0] = 0
    volumeneg = simpsonintegrate(fieldneg, x_cell_size, y_cell_size)

    s2 = ok.sigma.reshape(xx.shape)
    np.nan_to_num(s2, copy=False, nan=0)
    # volume_error = simpsonintegrate(s2, x_cell_size, y_cell_size)

    if plot_contours:
        plot_contour_krig(df, xx, yy, fieldpos, x, y)
    if plot_grid:
        plot_heatmap_krig(xx, yy, fieldpos)

    print(
        f'The emissions flux is {volume:.3f}kgh⁻¹; '
        f'the cut and fill volumes of the grid are {volumepos:.3f} and {volumeneg:.3f}kgh⁻¹. '
        f'The grid itself is {x_nodes}x{y_nodes} nodes, with each node measuring {x_cell_size:.2f}m x {y_cell_size:.2f}m.'
    )
