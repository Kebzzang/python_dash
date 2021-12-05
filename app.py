import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
# Interactive Visualliztion Lib
import plotly.express as px
import plotly.graph_objects as go
# Data Frame Lib 
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# Data Load
df1 = pd.read_csv('Raw_Data_Fin (1).csv', encoding='cp949')
# Pivot table
df_path = pd.read_csv('Path_Error_Rate_Table.csv')
# Graph Plot
fig1 = px.bar(df_path, x='Path', y='Error_Chip_Rate')

Error_Chip_Rate = df_path['Error_Chip_Rate'][0] * 100
Error_Chip_Top_Path = df_path['Path'][0]
df_pathes = pd.read_csv('Wafer_Path_Info.csv')
# fig2=px.bar(df_pathes, x='PATH',y='Error_Chip_Rate_Path')
# Card Layout 0

# 웨이퍼 맵 그리기 => 데이터 형 변환 2차원 리스트로 변형 => 최악의 수율 웨이퍼를 그려주기

# df1의 Target이 가장 많은 것 고르기
# df1['']
worst_target = df1.iloc[df1['Target'].idxmax()]
worst_target_idx = df1['Target'].idxmax()
print("최악의 수율 index: {}".format(worst_target))
wafer16 = df1.iloc[worst_target_idx]['Wafer_map']
print("check: {}, {}, {}, {}, {}, {}".format(worst_target['Thin F4'], worst_target['Temp_OXid'], worst_target['ppm'],
                                             worst_target['spin2'], worst_target['spin3'],
                                             worst_target['Source_Power']))


def wafer_map(wafer):
    wafer_list = []

    die = []
    k = 0
    for i in wafer:
        if i in ['0', '1', '2']:
            die.append(int(i))
        if len(die) == 26:
            wafer_list.append(die)
            die = []

    return px.imshow(wafer_list)


fig2 = wafer_map(wafer16)
cards = [
    dbc.Card([html.H2("2021년 11월 19일"),

              html.Div([dbc.Badge("단일 경로의 불량", color="danger", className="me-1"),
                        html.Div(f"{Error_Chip_Top_Path}", className='card_text')], style={'display': 'flex'}),
              html.H5(f"{Error_Chip_Rate}%", className='card_title'),

              ], body=True, color="PowerBlue")
]

#######################################################################################################
# Model Prediction 
from dash.dependencies import Input, Output
import pickle

# Model Load
model = pickle.load(open('model_RF.sav', 'rb'))
model_RF = pickle.load(open('GBmodel2.sav', 'rb'))


# Call Back (Dynamic)
@app.callback(
    Output('result', 'children'),
    Input('ppm', 'value'),
    Input('Thin F4', 'value'),
    Input('Temp_OXid', 'value'),
    Input('Source_Power', 'value'),
    Input('spin2', 'value'),
    Input('spin3', 'value')

)
def predict_func(x1, x2, x3, x4, x5, x6):
    list1 = [x1, x2, x3, x4, x5, x6]
    list_ = ['ppm', 'Thin F4', 'Temp_OXid', 'Source_Power', 'spin2', 'spin3']
    list_colname = ['Temp_OXid', 'ppm', 'Thin F4', 'spin2', 'spin3', 'Source_Power']
    input_data = dict(zip(list_, list1))
    input_X = pd.DataFrame(data=[list1], columns=input_data)
    print(input_X)
    print("모델:", type(model_RF))
    #     print("모델 스코어: ", model_RF.accuracy_score)

    prob_value = model_RF.predict(input_X)
    result = prob_value
    # 만약 그럴 가능성으로 찍고 싶으면 predict_proba()
    if result[0]==0:
        return "해당 조건의 웨이퍼는 양품입니다."
    else:
        return "해당 조건의 웨이퍼는 불량품입니다."

# Input Components Layout
input_layout = [

    html.Div([
        dbc.Label('씬 필름4'),
        dbc.Input(id='Thin F4', placeholder="씬 필름 두께", type='number'),
    ]),
    html.Div([
        dbc.Label('산화 공정 시 온도'),
        dbc.Input(id='Temp_OXid', placeholder='공정이 수행되께는 동안 Chamber 내 평균 온도 ', type='number')
    ]),
    html.Div([
        dbc.Label('ppm'),
        dbc.Input(id='ppm', placeholder='공정에 투여되는 합성물의 량 (ppm 단위)', type='number')
    ]),
    html.Div([
        dbc.Label('스핀2'),
        dbc.Input(id='spin2', placeholder='Spin Coat 과정에서 두 번째 회전 스핀 수 (rpm 단위)', type='number')
    ]),
    html.Div([
        dbc.Label('스핀3'),
        dbc.Input(id='spin3', placeholder='Spin Coat 과정에서 세 번째 회전 스핀 수 (rpm 단위)', type='number')
    ]),

    html.Div([
        dbc.Label('소스파워'),
        dbc.Input(id='Source_Power', placeholder='드라이 엣칭 공정 플라즈마 소스 파워', type='number')
    ]),
]

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("생산모니터링", href="#")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("logout", header=True),
                dbc.DropdownMenuItem("직원관리", href="#"),
                dbc.DropdownMenuItem("메신저", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="Admin",
        ),
    ],
    brand="TOSCO 반도체 생산관리",
    brand_href="#",
    color="dark",
    dark=True,
)
# Main Layout 
app.layout = dbc.Container(
    [
        navbar,

        html.Br(),
        dbc.Col(html.Div(cards)),  # Card  Layout
        dbc.Col(children=[html.Div(dcc.Graph(figure=fig1))]),  # Graph
        # worst_target
        html.H4(
            children=f"최악 수율 웨이퍼-{worst_target['Target']}을 기록한 {worst_target['No_Die']}번 웨이퍼의 에러메시지 :{worst_target['Error_message']}"),
        # 제목
        dbc.Col(children=[html.Div(dcc.Graph(figure=fig2))]),
        html.Hr(),  # 밑줄
        dbc.Col(html.Div(input_layout)),  # Input Layout
        dbc.Card([
            html.H2(id='result', className='card_title'),
            html.P('주어진 조건으로 ML Predict', className='card_text')
        ], body=True),
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
