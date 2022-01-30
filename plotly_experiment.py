from plotly.subplots import make_subplots
import plotly.graph_objs as go

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(x=[1,2,3], y=[4,5,6]),row=1, col=1)

fig.add_trace(go.Scatter(x=[20,30,40], y=[50,60,70]), row=1, col=2)

fig.update_layout(height=600, width=800, title_text="Example")
fig.show()