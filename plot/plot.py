import plotly.express as px
import plotly.graph_objects as go
import base64
import math
import pandas as pd
import numpy as np
from plot.plot_preparation import prepare_dataframe, group_expression_level, group_expression, group_interest_area


def readtxt(file_path):
    with open(file_path) as f:
        return eval(f.read())


def plot_graph(graph_type):

    expression_data = readtxt('./export/temp/expression.txt')
    behavior_data = readtxt('./export/temp/behavior.txt')
    image_data = readtxt('./export/temp/image.txt')
    interest_area_data = readtxt('./export/temp/interest_area.txt')

    level_orders = ['positive', 'neutral', 'negative']

    level_colors = {
        'negative': '#fc6600',
        'neutral': 'rgb(215,215,215)',
        'positive': '#4cbb17'
    }

    expression_colors = {
        'disgust': '#ff2400',
        'fear': '#fc6600',
        'anger': '#fda50f',
        'sadness': '#fed46e',
        'contempt': '#fced81',
        'neutral': 'rgb(215,215,215)',
        'surprise': '#a1bd57',
        'happy': '#4cbb17'
    }

    pattern = {
        'positive_and_positive': '++',
        'positive_and_neutral': '+N',
        'positive_and_negative': '+-',
        'neutral_and_positive': 'N+',
        'neutral_and_neutral': 'NN',
        'neutral_and_negative': 'N-',
        'negative_and_positive': '-+',
        'negative_and_neutral': '-N',
        'negative_and_negative': '--',
    }

    rect_spacing = 0.35
    rect_count = 1

    dataframe = prepare_dataframe(expression_data=expression_data, behavior_data=behavior_data,
                                  image_data=image_data, interest_area_data=interest_area_data)
    dataframe_interest_area_group = group_interest_area(
        dataframe=dataframe)

    dfs, overview_result = calculate_expression_pattern(
        dataframe_interest_area_group)

    fig = go.Figure()

    if graph_type == 'expression':
        dataframe_expression = group_expression(
            dataframe=dataframe)
        fig.add_traces(px.scatter(dataframe_expression,
                                  x=dataframe_expression.index,
                                  y='wrap',
                                  color='expression',
                                  color_discrete_map=expression_colors,
                                  custom_data=['expression',
                                               'frame', 'behavior'],
                                  ).data
                       )

        fig.update_xaxes(rangemode="nonnegative")

    else:
        dataframe_expression_level = group_expression_level(
            dataframe=dataframe)
        fig.add_traces(px.scatter(dataframe_expression_level,
                                  x=dataframe_expression_level.index,
                                  y='wrap',
                                  color='level',
                                  category_orders={  # replaces default order by column name
                                      "level": level_orders
                                  },
                                  color_discrete_map=level_colors,
                                  custom_data=['expression',
                                               'frame', 'behavior'],
                                  ).data
                       )

    hover_template = """
    <b>%{customdata[0]}</b><br>
    <b>Index:</b> %{x}<br>
    <b>Frame:</b> %{customdata[1]}<br>
    <b>Behavior:</b> %{customdata[2]}
    <extra></extra>
    """

    fig.update_traces(hovertemplate=hover_template)

    fig.update_traces(marker=dict(
        line=dict(width=1.5), size=14, symbol='142'))

    # draw interest area by rectangle index
    for trace in dfs:

        if trace['area'].iloc[0] >= 0:

            x_start = trace['area'].head(1).index.item()
            x_end = trace['area'].tail(1).index.item()
            y_bottom = 0.3
            y_top = 0.7

            # draw rectangle
            fig.add_vrect(
                xref="x",
                yref="paper",
                x0=x_start-rect_spacing,
                x1=x_end+rect_spacing,
                y0=y_bottom,
                y1=y_top
            )

            # draw annotation text
            fig.add_annotation(
                xref="x",
                yref="paper",
                x=(x_start+x_end)/2,
                y=y_top,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                text=f"IA {rect_count}"
            )

            # draw annotation text (pattern result)
            fig.add_annotation(
                xref="x",
                yref="paper",
                x=(x_start+x_end)/2,
                y=y_bottom,
                xanchor="center",
                yanchor="top",
                showarrow=False,
                text=pattern[trace['pattern'].iloc[0]]
            )

            # draw image
            eating_image = base64.b64encode(
                open(trace['image'].iloc[0], 'rb').read()
            )

            fig.add_layout_image(
                dict(
                    source='data:image/png;base64,{}'.format(
                        eating_image.decode()
                    ),
                    xref="x",
                    yref="paper",
                    yanchor="bottom",
                    x=x_start+0.75,
                    y=0.525,
                    sizex=26,
                    sizey=1
                )
            )

            rect_count = rect_count + 1

    fig.update_yaxes(showticklabels=False)

    fig.update_layout(
        title="Emotion Result per Frame",
        xaxis_title="Frame",
        yaxis_title="Emotion",
        legend_title="Emotion Level",
    )

    overview_template = f"""
    <b>Overview Result (Interest)</b><br>
    <b>Positive:</b> {overview_result.loc[overview_result.level == 'positive', 'count'].values[0]} frames<br>
    <b>Neutral:</b> {overview_result.loc[overview_result.level == 'neutral', 'count'].values[0]} frames<br>
    <b>Negative:</b> {overview_result.loc[overview_result.level == 'negative', 'count'].values[0]} frames
    """
    # draw annotation text (overview result)
    fig.add_annotation(
        xref="x",
        yref="paper",
        x=0,
        y=1,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        text=overview_template,
        font=dict(color="#000"),
        borderpad=8,
        bgcolor="#ff7f0e",
        opacity=0.8
    )

    fig.update_layout(template='plotly_white')

    fig.show()


def calculate_expression_pattern(_dataframe):
    init_level = pd.DataFrame(
        data={'level': ['positive', 'neutral', 'negative']})

    data = [{'level': 'positive', 'count': 0}, {
        'level': 'neutral', 'count': 0}, {'level': 'negative', 'count': 0}]
    overview_result = pd.DataFrame(data=data, columns=['level', 'count'])

    dfs = []
    for index, dataframe in _dataframe:

        dataframe["timeline"] = np.nan
        dataframe["pattern"] = np.nan

        if dataframe['area'].iloc[0] >= 0:
            # center = math.ceil((dataframe.index[-1] + dataframe.index[0])/2)
            center = math.ceil(((dataframe.index[-1] - dataframe.index[0])*(0.6))+dataframe.index[0])

            dataframe.loc[(dataframe.index < center), "timeline"] = "first"
            dataframe.loc[(dataframe.index >= center), "timeline"] = "last"

            sum_all = dataframe.groupby('level')['timeline'].agg([
                'count']).reset_index()
            sum_first = dataframe.groupby('level')['timeline'].agg(
                lambda x: (x == 'first').sum()).reset_index()
            sum_last = dataframe.groupby('level')['timeline'].agg(
                lambda x: (x == 'last').sum()).reset_index()


            result_all = pd.merge(sum_last, init_level, on="level", how="right").fillna(
                0).sort_values(by=['timeline'], ascending=False)
            result_all.columns = ['level','count']
                
            result_first = pd.merge(sum_first, init_level, on="level", how="right").fillna(
                0).sort_values(by=['timeline'], ascending=False).head(1)
            result_last = pd.merge(sum_last, init_level, on="level", how="right").fillna(
                0).sort_values(by=['timeline'], ascending=False).head(1)

            overview_result.loc[(overview_result.level == 'positive'), "count"] += math.ceil(
                result_all.loc[(result_all.level == 'positive'), "count"])
            overview_result.loc[(overview_result.level == 'neutral'), "count"] += math.ceil(
                result_all.loc[(result_all.level == 'neutral'), "count"])
            overview_result.loc[(overview_result.level == 'negative'), "count"] += math.ceil(
                result_all.loc[(result_all.level == 'negative'), "count"])

            result_final = f"{result_first.iloc[0]['level']}_and_{result_last.iloc[0]['level']}"
            dataframe['pattern'] = result_final

        dfs.append(dataframe)

    return dfs, overview_result
