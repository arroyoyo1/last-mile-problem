from shiny import ui

app_ui = ui.page_fluid(
    ui.h2("Mi Shiny App"),
    ui.input_slider("n", "Número de observaciones", 0, 100, 20),
    ui.output_text_verbatim("txt"),
    ui.output_plot("plot")
)