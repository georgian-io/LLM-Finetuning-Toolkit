from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.table import Table


def inject_example_to_rich_layout(layout: Layout, layout_name: str, example: dict):
    example = example.copy()

    # Crate Table
    table = Table(expand=True)
    colors = [
        "navy_blue",
        "dark_green",
        "spring_green3",
        "turquoise2",
        "cyan",
        "blue_violet",
        "royal_blue1",
        "steel_blue1",
        "chartreuse1",
        "deep_pink4",
        "plum2",
        "red",
    ]

    # Crate Formatted Text
    formatted = example.pop("formatted_prompt", None)
    formatted_text = Text(formatted)

    for key, c in zip(example.keys(), colors):
        table.add_column(key, style=c)

        tgt_text = example[key]
        start_idx = formatted.find(tgt_text)
        formatted_text.stylize(f"bold {c}", start_idx, start_idx + len(tgt_text))

    table.add_row(*example.values())

    layout.split_column(
        Layout(
            Panel(
                table,
                title=f"{layout_name} - Raw",
                title_align="left",
            )
        ),
        Layout(
            Panel(
                formatted_text,
                title=f"{layout_name} - Formatted",
                title_align="left",
            )
        ),
    )
