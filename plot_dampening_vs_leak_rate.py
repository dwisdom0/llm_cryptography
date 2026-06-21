import plotly.express as px

wide_data = {
    "dampening": [80, 82, 85, 88, 90, 92, 95, 99],
    "refusal_rate": [95.4, 83, 60, 31.2, 16.7, 3, 0, 0],
    "leak_rate": [2.3, 7, 13.7, 15.9, 8.4, 0, 0, 0],
}

# actually I need to make it long format for easy plotting
data = {
    "dampening": wide_data["dampening"] + wide_data["dampening"],
    "name": (["Refusal rate"] * len(wide_data["dampening"]))
    + (["Leak rate"] * len(wide_data["dampening"])),
    "rate": wide_data["refusal_rate"] + wide_data["leak_rate"],
}

fig = px.line(
    data,
    x="dampening",
    y="rate",
    color="name",
    title="As dampening increases, the model leaks the secret and then collapses into gibberish",
    # labels={'refusal_rate': 'Refusal rate', 'leak_rate': 'Leak rate'}, # this doesn't seem to work
)
fig.update_layout(
    legend=dict(title_text=""),
    xaxis=dict(title_text="Dampening percentage (higher = less signal propogated)"),
    yaxis=dict(title_text="Percentage of outputs (n=1,000)"),
)
fig.show()

fig.write_html(
    "plots/dampening_vs_leak_rate.html", include_plotlyjs=False, full_html=False
)
