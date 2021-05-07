import plotly.express as px
import plotly.graph_objects as go
import base64
from plot.plot_preparation import prepare_dataframe, group_expression_level, group_interest_area


def plot_graph(expression_data, behavior_data, image_data, interest_area_data):

    level_orders = ['positive', 'neutral', 'negative']

    level_colors = {'negative': '#fc6600',
                    'neutral': 'rgb(215,215,215)',
                    'positive': '#4cbb17'
                    }

    rect_spacing = 0.35

    dataframe = prepare_dataframe(expression_data=expression_data, behavior_data=behavior_data,
                                  image_data=image_data, interest_area_data=interest_area_data)
    dataframe_expression_level = group_expression_level(dataframe=dataframe)
    dataframe_interest_area_group = group_interest_area(dataframe=dataframe)

    fig = go.Figure()

    fig.add_traces(px.scatter(dataframe_expression_level,
                              x=dataframe_expression_level.index,
                              y='wrap',
                              color='level',
                              category_orders={  # replaces default order by column name
                                "level": level_orders
                              },
                              color_discrete_map=level_colors,
                              custom_data=['expression', 'frame', 'behavior'],
                              ).data
                   )

    # draw interest area by rectangle index
    dfs = []
    for name, data in dataframe_interest_area_group:
        dfs.append(data)

    for trace in dfs:
        if trace['area'].iloc[0] == 'interest':
            fig.add_vrect(x0=trace['area'].head(1).index.item()-rect_spacing,
                          x1=trace['area'].tail(1).index.item()+rect_spacing,
                          annotation_text="Interest Area", annotation_position="outside top")

            eating_image = base64.b64encode(
                open(trace['image'].iloc[0], 'rb').read())
            fig.add_layout_image(
                dict(
                    source='data:image/png;base64,{}'.format(
                        eating_image.decode()),
                    xref="x",
                    yref="y",
                    x=trace['behavior'].head(1).index.item(),
                    y=0.98,
                    sizex=16,
                    sizey=3,
                )
            )

    template = """
    <b>%{customdata[0]}</b><br>
    <b>Index:</b> %{x}<br>
    <b>Frame:</b> %{customdata[1]}<br>
    <b>Behavior:</b> %{customdata[2]}
    <extra></extra>
    """
    fig.update_traces(hovertemplate=template)

    fig.update_traces(marker=dict(line=dict(width=4), size=14, symbol='142'))

    fig.update_yaxes(showticklabels=False)

    fig.update_layout(
        title="Emotion Result per Frame",
        xaxis_title="Frame",
        yaxis_title="Emotion",
        legend_title="Emotion Level",
    )

    fig.update_layout(template='plotly_white')

    fig.show()


# if __name__ == "__main__":
#     expression_data = [[0.0, 'neutral'], [5.0, 'neutral'], [10.0, 'disgust'], [15.0, 'neutral'], [20.0, 'neutral'], [25.0, 'disgust'], [30.0, 'neutral'], [35.0, 'neutral'], [40.0, 'disgust'], [45.0, 'neutral'], [50.0, 'neutral'], [55.0, 'neutral'], [60.0, 'neutral'], [65.0, 'neutral'], [70.0, 'neutral'], [75.0, 'neutral'], [80.0, 'neutral'], [85.0, 'neutral'], [90.0, 'disgust'], [95.0, 'neutral'], [96.0, 'neutral'], [97.0, 'surprise'], [98.0, 'surprise'], [99.0, 'surprise'], [100.0, 'neutral'], [101.0, 'neutral'], [102.0, 'neutral'], [103.0, 'neutral'], [104.0, 'happy'], [105.0, 'happy'], [106.0, 'happy'], [107.0, 'happy'], [108.0, 'neutral'], [109.0, 'neutral'], [110.0, 'neutral'], [111.0, 'happy'], [112.0, 'happy'], [113.0, 'happy'], [114.0, 'happy'], [115.0, 'disgust'], [116.0, 'neutral'], [117.0, 'happy'], [118.0, 'happy'], [119.0, 'happy'], [120.0, 'disgust'], [121.0, 'disgust'], [122.0, 'disgust'], [123.0, 'disgust'], [124.0, 'neutral'], [125.0, 'disgust'], [126.0, 'disgust'], [127.0, 'disgust'], [128.0, 'disgust'], [129.0, 'neutral'], [130.0, 'neutral'], [131.0, 'disgust'], [132.0, 'neutral'], [133.0, 'neutral'], [134.0, 'neutral'], [135.0, 'neutral'], [136.0, 'disgust'], [137.0, 'neutral'], [138.0, 'disgust'], [139.0, 'neutral'], [140.0, 'neutral'], [141.0, 'neutral'], [142.0, 'neutral'], [143.0, 'neutral'], [144.0, 'neutral'], [145.0, 'neutral'], [150.0, 'neutral'], [155.0, 'disgust'], [160.0, 'disgust'], [165.0, 'neutral'], [170.0, 'disgust'], [175.0, 'disgust'], [180.0, 'disgust'], [185.0, 'disgust'], [190.0, 'neutral'], [195.0, 'happy'], [200.0, 'happy'], [205.0, 'disgust'], [
#         210.0, 'neutral'], [215.0, 'disgust'], [220.0, 'disgust'], [225.0, 'disgust'], [230.0, 'neutral'], [235.0, 'neutral'], [240.0, 'neutral'], [245.0, 'neutral'], [250.0, 'neutral'], [255.0, 'disgust'], [260.0, 'neutral'], [265.0, 'neutral'], [270.0, 'disgust'], [275.0, 'neutral'], [280.0, 'neutral'], [281.0, 'neutral'], [282.0, 'neutral'], [283.0, 'happy'], [284.0, 'happy'], [285.0, 'happy'], [286.0, 'happy'], [287.0, 'happy'], [288.0, 'happy'], [289.0, 'happy'], [290.0, 'happy'], [291.0, 'happy'], [292.0, 'happy'], [293.0, 'happy'], [294.0, 'happy'], [295.0, 'happy'], [296.0, 'disgust'], [297.0, 'disgust'], [298.0, 'neutral'], [299.0, 'neutral'], [300.0, 'surprise'], [301.0, 'neutral'], [302.0, 'neutral'], [303.0, 'disgust'], [304.0, 'disgust'], [305.0, 'disgust'], [306.0, 'disgust'], [307.0, 'disgust'], [308.0, 'disgust'], [309.0, 'disgust'], [310.0, 'disgust'], [311.0, 'disgust'], [312.0, 'disgust'], [313.0, 'disgust'], [314.0, 'disgust'], [315.0, 'disgust'], [316.0, 'disgust'], [317.0, 'disgust'], [318.0, 'disgust'], [319.0, 'disgust'], [320.0, 'disgust'], [321.0, 'disgust'], [322.0, 'disgust'], [323.0, 'disgust'], [324.0, 'disgust'], [325.0, 'disgust'], [326.0, 'disgust'], [327.0, 'disgust'], [328.0, 'disgust'], [329.0, 'disgust'], [330.0, 'disgust'], [335.0, 'disgust'], [340.0, 'disgust'], [345.0, 'disgust'], [350.0, 'disgust'], [355.0, 'disgust'], [360.0, 'neutral'], [365.0, 'neutral'], [370.0, 'neutral'], [375.0, 'disgust'], [380.0, 'neutral'], [385.0, 'disgust'], [390.0, 'disgust'], [395.0, 'neutral'], [400.0, 'anger'], [405.0, 'anger'], [410.0, 'anger'], [415.0, 'anger'], [420.0, 'anger']]

#     behavior_data = [[0.0, 'noeat'], [5.0, 'noeat'], [10.0, 'noeat'], [15.0, 'noeat'], [20.0, 'noeat'], [25.0, 'noeat'], [30.0, 'noeat'], [35.0, 'noeat'], [40.0, 'noeat'], [45.0, 'noeat'], [50.0, 'noeat'], [55.0, 'noeat'], [60.0, 'noeat'], [65.0, 'noeat'], [70.0, 'noeat'], [75.0, 'noeat'], [80.0, 'noeat'], [85.0, 'noeat'], [90.0, 'noeat'], [95.0, 'eat'], [96.0, 'eat'], [97.0, 'eat'], [98.0, 'eat'], [99.0, 'eat'], [100.0, 'eat'], [101.0, 'eat'], [102.0, 'eat'], [103.0, 'eat'], [104.0, 'eat'], [105.0, 'eat'], [106.0, 'eat'], [107.0, 'eat'], [108.0, 'noeat'], [109.0, 'noeat'], [110.0, 'noeat'], [111.0, 'eat'], [112.0, 'eat'], [113.0, 'eat'], [114.0, 'eat'], [115.0, 'eat'], [116.0, 'noeat'], [117.0, 'noeat'], [118.0, 'noeat'], [119.0, 'noeat'], [120.0, 'noeat'], [121.0, 'noeat'], [122.0, 'noeat'], [123.0, 'noeat'], [124.0, 'noeat'], [125.0, 'noeat'], [126.0, 'noeat'], [127.0, 'noeat'], [128.0, 'noeat'], [129.0, 'noeat'], [130.0, 'noeat'], [131.0, 'noeat'], [132.0, 'noeat'], [133.0, 'noeat'], [134.0, 'noeat'], [135.0, 'noeat'], [136.0, 'noeat'], [137.0, 'noeat'], [138.0, 'noeat'], [139.0, 'noeat'], [140.0, 'noeat'], [141.0, 'noeat'], [142.0, 'noeat'], [143.0, 'noeat'], [144.0, 'noeat'], [145.0, 'noeat'], [150.0, 'noeat'], [155.0, 'noeat'], [160.0, 'noeat'], [165.0, 'noeat'], [170.0, 'noeat'], [175.0, 'noeat'], [180.0, 'noeat'], [185.0, 'noeat'], [190.0, 'noeat'], [195.0, 'noeat'], [200.0, 'noeat'], [205.0, 'noeat'], [210.0, 'noeat'], [
#         215.0, 'noeat'], [220.0, 'noeat'], [225.0, 'noeat'], [230.0, 'noeat'], [235.0, 'noeat'], [240.0, 'noeat'], [245.0, 'noeat'], [250.0, 'noeat'], [255.0, 'noeat'], [260.0, 'noeat'], [265.0, 'noeat'], [270.0, 'noeat'], [275.0, 'noeat'], [280.0, 'eat'], [281.0, 'eat'], [282.0, 'eat'], [283.0, 'eat'], [284.0, 'eat'], [285.0, 'eat'], [286.0, 'eat'], [287.0, 'eat'], [288.0, 'eat'], [289.0, 'eat'], [290.0, 'eat'], [291.0, 'eat'], [292.0, 'eat'], [293.0, 'noeat'], [294.0, 'noeat'], [295.0, 'noeat'], [296.0, 'noeat'], [297.0, 'noeat'], [298.0, 'noeat'], [299.0, 'noeat'], [300.0, 'noeat'], [301.0, 'noeat'], [302.0, 'noeat'], [303.0, 'noeat'], [304.0, 'noeat'], [305.0, 'noeat'], [306.0, 'noeat'], [307.0, 'noeat'], [308.0, 'noeat'], [309.0, 'noeat'], [310.0, 'noeat'], [311.0, 'noeat'], [312.0, 'noeat'], [313.0, 'noeat'], [314.0, 'noeat'], [315.0, 'noeat'], [316.0, 'noeat'], [317.0, 'noeat'], [318.0, 'noeat'], [319.0, 'noeat'], [320.0, 'noeat'], [321.0, 'noeat'], [322.0, 'noeat'], [323.0, 'noeat'], [324.0, 'noeat'], [325.0, 'noeat'], [326.0, 'noeat'], [327.0, 'noeat'], [328.0, 'noeat'], [329.0, 'noeat'], [330.0, 'noeat'], [335.0, 'noeat'], [340.0, 'noeat'], [345.0, 'noeat'], [350.0, 'noeat'], [355.0, 'noeat'], [360.0, 'noeat'], [365.0, 'noeat'], [370.0, 'noeat'], [375.0, 'noeat'], [380.0, 'noeat'], [385.0, 'noeat'], [390.0, 'noeat'], [395.0, 'noeat'], [400.0, 'noeat'], [405.0, 'noeat'], [410.0, 'noeat'], [415.0, 'noeat'], [420.0, 'noeat']]

#     image_data = [[95.0, '../export/export_first_frame_eat_95.0.jpg'],
#                   [280.0, '../export/export_first_frame_eat_280.0.jpg']]

#     interest_area_data = [[0.0, 5], [5.0, 5], [10.0, 5], [15.0, 5], [20.0, 5], [25.0, 5], [30.0, 5], [35.0, 5], [40.0, 5], [45.0, 5], [50.0, 5], [55.0, 5], [60.0, 5], [65.0, 5], [70.0, 5], [75.0, 5], [80.0, 5], [85.0, 5], [90.0, 5], [95.0, 1], [96.0, 1], [97.0, 1], [98.0, 1], [99.0, 1], [100.0, 1], [101.0, 1], [102.0, 1], [103.0, 1], [104.0, 1], [105.0, 1], [106.0, 1], [107.0, 1], [108.0, 1], [109.0, 1], [110.0, 1], [111.0, 1], [112.0, 1], [113.0, 1], [114.0, 1], [115.0, 1], [116.0, 1], [117.0, 1], [118.0, 1], [119.0, 1], [120.0, 1], [121.0, 1], [122.0, 1], [123.0, 1], [124.0, 1], [125.0, 1], [126.0, 1], [127.0, 1], [128.0, 1], [129.0, 1], [130.0, 1], [131.0, 1], [132.0, 1], [133.0, 1], [134.0, 1], [135.0, 1], [136.0, 1], [137.0, 1], [138.0, 1], [139.0, 1], [140.0, 1], [141.0, 1], [142.0, 1], [143.0, 1], [144.0, 5], [145.0, 5], [150.0, 5], [155.0, 5], [160.0, 5], [165.0, 5], [170.0, 5], [175.0, 5], [180.0, 5], [185.0, 5], [190.0, 5], [195.0, 5], [200.0, 5], [205.0, 5], [210.0, 5], [
#         215.0, 5], [220.0, 5], [225.0, 5], [230.0, 5], [235.0, 5], [240.0, 5], [245.0, 5], [250.0, 5], [255.0, 5], [260.0, 5], [265.0, 5], [270.0, 5], [275.0, 5], [280.0, 1], [281.0, 1], [282.0, 1], [283.0, 1], [284.0, 1], [285.0, 1], [286.0, 1], [287.0, 1], [288.0, 1], [289.0, 1], [290.0, 1], [291.0, 1], [292.0, 1], [293.0, 1], [294.0, 1], [295.0, 1], [296.0, 1], [297.0, 1], [298.0, 1], [299.0, 1], [300.0, 1], [301.0, 1], [302.0, 1], [303.0, 1], [304.0, 1], [305.0, 1], [306.0, 1], [307.0, 1], [308.0, 1], [309.0, 1], [310.0, 1], [311.0, 1], [312.0, 1], [313.0, 1], [314.0, 1], [315.0, 1], [316.0, 1], [317.0, 1], [318.0, 1], [319.0, 1], [320.0, 1], [321.0, 1], [322.0, 1], [323.0, 1], [324.0, 1], [325.0, 1], [326.0, 1], [327.0, 1], [328.0, 1], [329.0, 5], [330.0, 5], [335.0, 5], [340.0, 5], [345.0, 5], [350.0, 5], [355.0, 5], [360.0, 5], [365.0, 5], [370.0, 5], [375.0, 5], [380.0, 5], [385.0, 5], [390.0, 5], [395.0, 5], [400.0, 5], [405.0, 5], [410.0, 5], [415.0, 5], [420.0, 5]]

#     plot_graph(expression_data, behavior_data, image_data, interest_area_data)
