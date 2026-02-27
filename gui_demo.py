"""
Neural-Network-RL-Dashboard — DearPyGUI demo.
Showcases windows, menus, inputs, sliders, buttons, plot, table, logging, and themes.
"""
import math
import dearpygui.dearpygui as dpg

# Tag IDs for widgets we update from callbacks
LOG_TEXT_TAG = "log_text"
CLICK_COUNT_TAG = "click_count"
PLOT_SERIES_TAG = "reward_series"

click_count = [0]  # use list so callback can mutate


def log_message(msg: str) -> None:
    """Append a line to the log window."""
    try:
        dpg.set_value(LOG_TEXT_TAG, dpg.get_value(LOG_TEXT_TAG) + msg + "\n")
    except Exception:
        pass


def on_run_click(sender, app_data, user_data) -> None:
    click_count[0] += 1
    dpg.set_value(CLICK_COUNT_TAG, str(click_count[0]))
    log_message(f"Run #{click_count[0]} — (demo: no training executed)")


def on_theme_dark(sender, app_data, user_data) -> None:
    dpg.bind_theme(user_data)
    log_message("Theme: Dark applied.")


def on_theme_light(sender, app_data, user_data) -> None:
    dpg.bind_theme(user_data)
    log_message("Theme: Light applied.")


def on_clear_log(sender, app_data, user_data) -> None:
    dpg.set_value(LOG_TEXT_TAG, "")
    log_message("Log cleared.")


def on_slider_change(sender, app_data, user_data) -> None:
    log_message(f"Slider '{dpg.get_item_label(sender)}' = {app_data}")


def build_main_window() -> int:
    with dpg.window(label="Neural-Network-RL-Dashboard (DearPyGUI Demo)", tag="main_window") as main_win:
        # --- Menu bar ---
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Clear log", callback=on_clear_log)
                dpg.add_menu_item(label="Exit", callback=lambda *_: dpg.stop_dearpygui())
            with dpg.menu(label="View"):
                dpg.add_menu_item(label="Dark theme", callback=on_theme_dark, user_data="theme_dark")
                dpg.add_menu_item(label="Light theme", callback=on_theme_light, user_data="theme_light")
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="About", callback=lambda *_: log_message("DearPyGUI demo for RL Dashboard."))

        # --- Top row: controls ---
        with dpg.group(horizontal=True):
            dpg.add_button(label="Run (demo)", callback=on_run_click)
            dpg.add_text("Clicks: ")
            dpg.add_text("0", tag=CLICK_COUNT_TAG)
            dpg.add_spacer(width=20)
            dpg.add_input_text(label="Config", default_value="default_config", tag="config_input")
            dpg.add_slider_float(
                label="Learning rate",
                default_value=1e-3,
                min_value=1e-5,
                max_value=1e-1,
                format="%.0e",
                callback=on_slider_change,
            )
            dpg.add_checkbox(label="Enable logging", default_value=True, tag="logging_cb")

        dpg.add_separator()

        # --- Two columns: plot + table | log ---
        with dpg.group(horizontal=True):
            # Left: plot and table
            with dpg.child_window(width=520, border=True):
                dpg.add_text("Reward curve (demo data)", color=(100, 200, 255))
                # Plot: dummy reward curve
                xs = [i * 0.5 for i in range(61)]
                ys = [10 * math.sin(i / 8) + 0.3 * i + 15 for i in range(61)]
                with dpg.plot(label="Reward", height=220, width=480):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Reward")
                    dpg.add_line_series(xs, ys, label="Reward", parent=y_axis, tag=PLOT_SERIES_TAG)
                    dpg.add_plot_legend()

                dpg.add_spacer(height=10)
                dpg.add_text("Episode summary (demo)", color=(100, 200, 255))
                with dpg.table(header_row=True, row_background=True, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
                    dpg.add_table_column(label="Episode")
                    dpg.add_table_column(label="Reward")
                    dpg.add_table_column(label="Length")
                    for i in range(1, 6):
                        with dpg.table_row():
                            dpg.add_text(str(i))
                            dpg.add_text(f"{10.0 + i * 2.5:.1f}")
                            dpg.add_text(str(100 + i * 10))

            # Right: log
            with dpg.child_window(border=True):
                dpg.add_text("Log", color=(100, 200, 255))
                dpg.add_input_text(
                    tag=LOG_TEXT_TAG,
                    multiline=True,
                    readonly=True,
                    default_value="Ready. Use Run (demo) or menus.\n",
                )

    return main_win


def build_themes() -> None:
    with dpg.theme(tag="theme_dark"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 120, 180))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 140, 220))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 100, 160))

    with dpg.theme(tag="theme_light"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 220, 240))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (220, 235, 252))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (180, 205, 230))


def main() -> None:
    dpg.create_context()
    build_themes()
    dpg.bind_theme("theme_dark")
    main_win = build_main_window()

    dpg.create_viewport(title="Neural-Network-RL-Dashboard — DearPyGUI Demo", width=900, height=580)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    log_message("DearPyGUI demo started. Try menus, Run button, and sliders.")

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
