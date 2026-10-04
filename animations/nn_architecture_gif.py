from pathlib import Path

from manim import *


# ================================================================
# FER-2013 BASELINE ARCHITECTURE ANIMATION
# ================================================================
#
# Two scenes are included:
#
#   1. NNArchitecturePresentation
#      - slower, more cinematic
#      - designed for a 16:9 presentation slide / MP4
#
#   2. NNArchitectureNotebook
#      - slightly tighter pacing
#      - designed for embedding as a GIF in the notebook
#
# Render from the repository root:
#
# Presentation:
#   python -m manim -pqh animations\nn_architecture_gif.py NNArchitecturePresentation
#
# Notebook:
#   python -m manim -pqh animations\nn_architecture_gif.py NNArchitectureNotebook
#
# The script automatically searches:
#   data/final/
#
# for a real FER image. If none is found, it falls back to a
# stylized grayscale pixel face.
# ================================================================


class ArchitectureBase(Scene):

    # ------------------------------------------------------------
    # Presentation-friendly palette
    # ------------------------------------------------------------

    BG = "#070A0F"
    TEXT = "#F4F4F4"
    MUTED = "#BFC5CE"

    FORWARD = "#EAECEF"
    GRADIENT = "#FF765D"
    UPDATE = "#5CC8FF"
    LOSS = "#F7C85B"
    SOFTMAX = "#A98BFF"

    CLASS_COLORS = {
        "A": "#FF6B6B",
        "H": "#70D6A3",
        "S": "#6EA8FE",
        "N": "#C8B6FF",
    }

    CLASS_NAMES = {
        "A": "angry",
        "H": "happy",
        "S": "sad",
        "N": "neutral",
    }

    # Overridden by child scenes.
    NOTEBOOK_MODE = False

    def construct(self):
        self.camera.background_color = self.BG

        if self.NOTEBOOK_MODE:
            reveal_time = 0.65
            hold = 0.35
            long_hold = 0.75
            flow_time = 0.55
        else:
            reveal_time = 0.90
            hold = 0.55
            long_hold = 1.15
            flow_time = 0.80

        # ========================================================
        # TITLE AREA
        # ========================================================

        title = Text(
            "FER-2013 Neural Network Baseline",
            font_size=34,
            weight="BOLD",
            color=self.TEXT,
        ).to_edge(UP, buff=0.20)

        subtitle = Text(
            "NumPy MLP  |  48x48 grayscale  |  4 emotion classes",
            font_size=18,
            color=self.MUTED,
        ).next_to(
            title,
            DOWN,
            buff=0.08,
        )

        phase_label = Text(
            "1. Forward pass",
            font_size=24,
            weight="BOLD",
            color=self.TEXT,
        ).next_to(
            subtitle,
            DOWN,
            buff=0.24,
        )

        self.play(
            FadeIn(
                title,
                shift=DOWN * 0.10,
            ),
            FadeIn(
                subtitle,
                shift=DOWN * 0.10,
            ),
            run_time=reveal_time,
        )

        self.play(
            FadeIn(
                phase_label
            ),
            run_time=0.50,
        )

        # ========================================================
        # NETWORK BLOCKS
        # ========================================================

        input_block = self.make_input_block()

        flatten_block = self.make_simple_block(
            "Flatten",
            "48x48 -> 2304",
            width=1.80,
            height=1.32,
        )

        hidden_block = self.make_hidden_dense_block()

        relu_block = self.make_simple_block(
            "ReLU",
            "max(0, x)",
            width=1.45,
            height=1.32,
        )

        output_block = self.make_output_dense_block()

        softmax_block = self.make_simple_block(
            "Softmax",
            "logits -> probs",
            width=1.65,
            height=1.32,
            title_color=self.SOFTMAX,
            border_color=self.SOFTMAX,
        )

        main_row = Group(
            input_block,
            flatten_block,
            hidden_block,
            relu_block,
            output_block,
            softmax_block,
        )

        main_row.arrange(
            RIGHT,
            buff=0.43,
        )

        main_row.scale_to_fit_width(
            config.frame_width - 0.65
        )

        main_row.next_to(
            phase_label,
            DOWN,
            buff=0.58,
        )

        # Keep room beneath the row for probability bars / epoch strip.
        main_row.shift(UP * 0.05)

        # ========================================================
        # CLASS COLOR LEGEND
        # ========================================================

        legend = self.make_class_legend()

        legend.to_corner(
            UR,
            buff=0.30,
        )

        legend.shift(
            DOWN * 0.95
        )

        # ========================================================
        # FORWARD ARROWS
        # ========================================================

        forward_arrows = VGroup(
            self.connect_right(
                input_block,
                flatten_block,
                self.FORWARD,
            ),
            self.connect_right(
                flatten_block,
                hidden_block,
                self.FORWARD,
            ),
            self.connect_right(
                hidden_block,
                relu_block,
                self.FORWARD,
            ),
            self.connect_right(
                relu_block,
                output_block,
                self.FORWARD,
            ),
            self.connect_right(
                output_block,
                softmax_block,
                self.FORWARD,
            ),
        )

        # ========================================================
        # FORWARD REVEAL
        # ========================================================

        self.play(
            FadeIn(
                input_block,
                shift=RIGHT * 0.10,
            ),
            run_time=reveal_time,
        )

        self.wait(hold)

        for arrow, block in zip(
            forward_arrows,
            main_row[1:],
        ):
            self.play(
                GrowArrow(
                    arrow
                ),
                FadeIn(
                    block,
                    shift=RIGHT * 0.10,
                ),
                run_time=reveal_time,
            )

            self.wait(hold)

        self.play(
            FadeIn(
                legend
            ),
            run_time=0.55,
        )

        # ========================================================
        # FORWARD DATA FLOW
        # ========================================================

        for arrow in forward_arrows:
            pulse = Dot(
                radius=0.06,
                color=YELLOW,
            ).move_to(
                arrow.get_start()
            )

            self.add(
                pulse
            )

            self.play(
                MoveAlongPath(
                    pulse,
                    arrow,
                ),
                run_time=flow_time,
                rate_func=linear,
            )

            self.remove(
                pulse
            )

        self.wait(hold)

        # ========================================================
        # SOFTMAX PROBABILITY BARS
        # ========================================================

        probability_panel = self.make_probability_panel()

        probability_panel.next_to(
            softmax_block,
            DOWN,
            buff=0.18,
        )

        self.play(
            FadeIn(
                probability_panel[0],
                shift=UP * 0.05,
            ),
            run_time=0.45,
        )

        # Rows reveal one by one.
        self.play(
            LaggedStart(
                *[
                    FadeIn(
                        row,
                        shift=RIGHT * 0.05,
                    )
                    for row in probability_panel[1]
                ],
                lag_ratio=0.12,
            ),
            run_time=1.00 if not self.NOTEBOOK_MODE else 0.75,
        )

        self.wait(long_hold)

        winner_box = SurroundingRectangle(
            probability_panel[1][1],
            buff=0.04,
            corner_radius=0.05,
            color=self.CLASS_COLORS["H"],
            stroke_width=2,
        )

        self.play(
            Create(
                winner_box
            ),
            run_time=0.55,
        )

        self.wait(long_hold)

        # ========================================================
        # LOSS
        # ========================================================

        loss_box = self.make_loss_box()

        loss_box.next_to(
            softmax_block,
            UP,
            buff=0.30,
        )

        softmax_to_loss = Arrow(
            softmax_block.get_top(),
            loss_box.get_bottom(),
            buff=0.08,
            stroke_width=3,
            color=self.LOSS,
            max_tip_length_to_length_ratio=0.16,
        )

        self.play(
            GrowArrow(
                softmax_to_loss
            ),
            FadeIn(
                loss_box,
                shift=DOWN * 0.05,
            ),
            run_time=reveal_time,
        )

        self.wait(long_hold)

        # ========================================================
        # BACKPROPAGATION
        # ========================================================

        backprop_title = Text(
            "2. Backpropagation",
            font_size=24,
            weight="BOLD",
            color=self.TEXT,
        ).move_to(
            phase_label
        )

        self.play(
            Transform(
                phase_label,
                backprop_title,
            ),
            FadeOut(
                probability_panel
            ),
            FadeOut(
                winner_box
            ),
            run_time=0.70,
        )

        self.wait(hold)

        # Clean gradient arrows. Labels sit centered above the path.
        loss_back = Arrow(
            loss_box.get_bottom(),
            softmax_block.get_top(),
            buff=0.08,
            stroke_width=3.2,
            color=self.GRADIENT,
            max_tip_length_to_length_ratio=0.16,
        )

        backward_arrows = VGroup(
            self.connect_left(
                softmax_block,
                output_block,
                self.GRADIENT,
                width=3.2,
            ),
            self.connect_left(
                output_block,
                relu_block,
                self.GRADIENT,
                width=3.2,
            ),
            self.connect_left(
                relu_block,
                hidden_block,
                self.GRADIENT,
                width=3.2,
            ),
            self.connect_left(
                hidden_block,
                flatten_block,
                self.GRADIENT,
                width=3.2,
            ),
        )

        grad_loss_label = self.gradient_badge(
            "loss gradient",
        ).next_to(
            loss_back,
            RIGHT,
            buff=0.10,
        )

        grad_labels = VGroup(
            self.gradient_badge(
                "dL / dz2"
            ),
            self.gradient_badge(
                "dL / da1"
            ),
            self.gradient_badge(
                "dL / dz1"
            ),
            self.gradient_badge(
                "input grad"
            ),
        )

        for label, arrow in zip(
            grad_labels,
            backward_arrows,
        ):
            label.next_to(
                arrow,
                UP,
                buff=0.08,
            )

        self.play(
            GrowArrow(
                loss_back
            ),
            FadeIn(
                grad_loss_label
            ),
            run_time=reveal_time,
        )

        self.wait(0.20)

        for arrow, label in zip(
            backward_arrows,
            grad_labels,
        ):
            pulse = Dot(
                radius=0.06,
                color=self.GRADIENT,
            ).move_to(
                arrow.get_start()
            )

            self.play(
                GrowArrow(
                    arrow
                ),
                FadeIn(
                    label
                ),
                run_time=reveal_time,
            )

            self.add(
                pulse
            )

            self.play(
                MoveAlongPath(
                    pulse,
                    arrow,
                ),
                run_time=flow_time,
                rate_func=linear,
            )

            self.remove(
                pulse
            )

        self.wait(long_hold)

        # ========================================================
        # PARAMETER GRADIENTS
        # ========================================================

        hidden_grad_badge = self.make_dense_gradient_badge(
            "weight gradients",
            "bias gradients",
        ).next_to(
            hidden_block,
            UP,
            buff=0.14,
        )

        output_grad_badge = self.make_dense_gradient_badge(
            "weight gradients",
            "bias gradients",
        ).next_to(
            output_block,
            UP,
            buff=0.14,
        )

        self.play(
            FadeIn(
                hidden_grad_badge,
                shift=DOWN * 0.05,
            ),
            FadeIn(
                output_grad_badge,
                shift=DOWN * 0.05,
            ),
            run_time=0.70,
        )

        self.wait(long_hold)

        # ========================================================
        # PARAMETER UPDATE
        # ========================================================

        update_title = Text(
            "3. Parameter update",
            font_size=24,
            weight="BOLD",
            color=self.TEXT,
        ).move_to(
            phase_label
        )

        self.play(
            Transform(
                phase_label,
                update_title,
            ),
            FadeOut(
                grad_loss_label
            ),
            FadeOut(
                grad_labels
            ),
            FadeOut(
                loss_back
            ),
            FadeOut(
                backward_arrows
            ),
            run_time=0.75,
        )

        # No formulas at the bottom.
        # Instead, the Dense blocks themselves are annotated.
        hidden_update_badge = self.make_update_badge(
            "update weights + bias"
        ).move_to(
            hidden_grad_badge
        )

        output_update_badge = self.make_update_badge(
            "update weights + bias"
        ).move_to(
            output_grad_badge
        )

        self.play(
            Transform(
                hidden_grad_badge,
                hidden_update_badge,
            ),
            Transform(
                output_grad_badge,
                output_update_badge,
            ),
            run_time=0.80,
        )

        self.play(
            Indicate(
                hidden_block[0],
                color=self.UPDATE,
                scale_factor=1.035,
            ),
            Indicate(
                output_block[0],
                color=self.UPDATE,
                scale_factor=1.035,
            ),
            run_time=1.20,
        )

        self.wait(long_hold)

        # ========================================================
        # TRAINING MODEL / MINI-EPOCH VISUAL
        # ========================================================

        training_title = Text(
            "4. Training",
            font_size=24,
            weight="BOLD",
            color=self.TEXT,
        ).move_to(
            phase_label
        )

        self.play(
            Transform(
                phase_label,
                training_title,
            ),
            FadeOut(
                hidden_grad_badge
            ),
            FadeOut(
                output_grad_badge
            ),
            FadeOut(
                softmax_to_loss
            ),
            FadeOut(
                loss_box
            ),
            run_time=0.75,
        )

        self.wait(hold)

        # The training strip intentionally has NO loop arrow.
        training_strip = self.make_training_strip()

        training_strip.next_to(
            main_row,
            DOWN,
            buff=0.32,
        )

        self.play(
            FadeIn(
                training_strip,
                shift=UP * 0.08,
            ),
            run_time=0.75,
        )

        self.wait(hold)

        # Animate each stage in order.
        strip_nodes = training_strip[0]

        for node in strip_nodes:
            self.play(
                Indicate(
                    node,
                    scale_factor=1.07,
                    color=YELLOW,
                ),
                run_time=0.48,
            )

        self.wait(hold)

        # Epoch progression: no fake loss/accuracy numbers.
        epoch_panel = self.make_epoch_panel()

        epoch_panel.next_to(
            training_strip,
            DOWN,
            buff=0.20,
        )

        self.play(
            FadeIn(
                epoch_panel[0]
            ),
            FadeIn(
                epoch_panel[1]
            ),
            run_time=0.60,
        )

        progress_bg = epoch_panel[2]
        progress_fill = epoch_panel[3]

        self.play(
            FadeIn(
                progress_bg
            ),
            FadeIn(
                progress_fill
            ),
            run_time=0.45,
        )

        # Grow from 0% to 100%.
        self.play(
            progress_fill.animate.stretch_to_fit_width(
                progress_bg.width
            ).align_to(
                progress_bg,
                LEFT,
            ),
            run_time=2.4 if not self.NOTEBOOK_MODE else 1.5,
            rate_func=linear,
        )

        self.wait(long_hold)

        # ========================================================
        # CLEAN FINAL FRAME
        # ========================================================

        final_title = Text(
            "Forward -> Loss -> Backprop -> Update -> Repeat",
            font_size=23,
            weight="BOLD",
            color=self.TEXT,
        ).move_to(
            phase_label
        )

        self.play(
            Transform(
                phase_label,
                final_title,
            ),
            run_time=0.65,
        )

        self.wait(
            2.2 if not self.NOTEBOOK_MODE else 1.2
        )

    # ================================================================
    # VISUAL HELPERS
    # ================================================================

    def make_input_block(self):
        frame = RoundedRectangle(
            width=1.75,
            height=1.95,
            corner_radius=0.12,
            stroke_width=2,
            color=WHITE,
        )

        title = Text(
            "Input",
            font_size=20,
            weight="BOLD",
            color=self.TEXT,
        ).next_to(
            frame,
            UP,
            buff=0.08,
        )

        visual = self.find_real_fer_image()

        if visual is not None:
            visual.set_height(
                1.20
            )

            visual.move_to(
                frame.get_center()
                + UP * 0.08
            )

        else:
            visual = self.make_pixel_grid()

            visual.set_height(
                1.15
            )

            visual.move_to(
                frame.get_center()
                + UP * 0.08
            )

        shape = Text(
            "48x48",
            font_size=15,
            color=self.MUTED,
        ).move_to(
            frame.get_center()
            + DOWN * 0.70
        )

        return Group(
            frame,
            title,
            visual,
            shape,
        )

    def make_simple_block(
        self,
        title,
        subtitle,
        width,
        height,
        title_color=None,
        border_color=None,
    ):
        title_color = (
            title_color
            if title_color is not None
            else self.TEXT
        )

        border_color = (
            border_color
            if border_color is not None
            else WHITE
        )

        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.10,
            stroke_width=2,
            color=border_color,
        )

        title_text = Text(
            title,
            font_size=20,
            weight="BOLD",
            color=title_color,
        ).move_to(
            box.get_center()
            + UP * 0.19
        )

        subtitle_text = Text(
            subtitle,
            font_size=14,
            color=self.MUTED,
        ).move_to(
            box.get_center()
            + DOWN * 0.27
        )

        return VGroup(
            box,
            title_text,
            subtitle_text,
        )

    def make_hidden_dense_block(self):
        box = RoundedRectangle(
            width=1.95,
            height=2.30,
            corner_radius=0.12,
            stroke_width=2,
            color=WHITE,
        )

        title = Text(
            "Dense",
            font_size=20,
            weight="BOLD",
            color=self.TEXT,
        ).next_to(
            box,
            UP,
            buff=0.08,
        )

        nodes = VGroup()

        for _ in range(6):
            nodes.add(
                Circle(
                    radius=0.075,
                    stroke_width=1.7,
                    color=self.GRADIENT,
                )
            )

        nodes.arrange(
            DOWN,
            buff=0.09,
        )

        nodes.move_to(
            box.get_center()
            + LEFT * 0.34
        )

        dots = Text(
            "...",
            font_size=17,
            color=self.MUTED,
        ).rotate(
            PI / 2
        )

        dots.move_to(
            box.get_center()
            + RIGHT * 0.05
        )

        count = Text(
            "x128",
            font_size=17,
            weight="BOLD",
            color=self.TEXT,
        ).move_to(
            box.get_center()
            + RIGHT * 0.42
        )

        foot = Text(
            "2304 -> 128",
            font_size=14,
            color=self.MUTED,
        ).next_to(
            box,
            DOWN,
            buff=0.08,
        )

        return VGroup(
            box,
            title,
            nodes,
            dots,
            count,
            foot,
        )

    def make_output_dense_block(self):
        box = RoundedRectangle(
            width=1.75,
            height=2.30,
            corner_radius=0.12,
            stroke_width=2,
            color=WHITE,
        )

        title = Text(
            "Dense",
            font_size=20,
            weight="BOLD",
            color=self.TEXT,
        ).next_to(
            box,
            UP,
            buff=0.08,
        )

        nodes = VGroup()

        for short_name in [
            "A",
            "H",
            "S",
            "N",
        ]:
            circle = Circle(
                radius=0.13,
                stroke_width=2,
                color=self.CLASS_COLORS[short_name],
            )

            letter = Text(
                short_name,
                font_size=15,
                weight="BOLD",
                color=self.CLASS_COLORS[short_name],
            ).move_to(
                circle
            )

            nodes.add(
                VGroup(
                    circle,
                    letter,
                )
            )

        nodes.arrange(
            DOWN,
            buff=0.10,
        )

        nodes.move_to(
            box
        )

        foot = Text(
            "128 -> 4 logits",
            font_size=14,
            color=self.MUTED,
        ).next_to(
            box,
            DOWN,
            buff=0.08,
        )

        return VGroup(
            box,
            title,
            nodes,
            foot,
        )

    def make_class_legend(self):
        rows = VGroup()

        for short_name in [
            "A",
            "H",
            "S",
            "N",
        ]:
            dot = Dot(
                radius=0.045,
                color=self.CLASS_COLORS[short_name],
            )

            label = Text(
                self.CLASS_NAMES[short_name],
                font_size=12,
                color=self.MUTED,
            )

            row = VGroup(
                dot,
                label,
            ).arrange(
                RIGHT,
                buff=0.07,
            )

            rows.add(
                row
            )

        rows.arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.04,
        )

        return rows

    def make_probability_panel(self):
        title = Text(
            "Example probabilities",
            font_size=14,
            weight="BOLD",
            color=self.TEXT,
        )

        class_order = [
            ("A", 0.18),
            ("H", 0.54),
            ("S", 0.09),
            ("N", 0.19),
        ]

        rows = VGroup()

        for short_name, probability in class_order:
            name = Text(
                self.CLASS_NAMES[short_name],
                font_size=11,
                color=self.MUTED,
            )

            bar_bg = Rectangle(
                width=1.05,
                height=0.11,
                stroke_width=0.8,
                color="#5A606B",
            )

            bar = Rectangle(
                width=max(
                    1.05 * probability,
                    0.035,
                ),
                height=0.11,
                stroke_width=0,
                fill_opacity=1,
                color=self.CLASS_COLORS[short_name],
            )

            bar.align_to(
                bar_bg,
                LEFT,
            )

            percent = Text(
                f"{probability:.0%}",
                font_size=11,
                color=self.TEXT,
            )

            row = VGroup(
                name,
                VGroup(
                    bar_bg,
                    bar,
                ),
                percent,
            ).arrange(
                RIGHT,
                buff=0.06,
            )

            rows.add(
                row
            )

        rows.arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.045,
        )

        return VGroup(
            title,
            rows,
        ).arrange(
            DOWN,
            buff=0.06,
        )

    def make_loss_box(self):
        box = RoundedRectangle(
            width=1.65,
            height=0.72,
            corner_radius=0.09,
            stroke_width=2,
            color=self.LOSS,
        )

        text = VGroup(
            Text(
                "Cross-Entropy",
                font_size=13,
                weight="BOLD",
                color=self.TEXT,
            ),
            Text(
                "Loss",
                font_size=15,
                weight="BOLD",
                color=self.LOSS,
            ),
        ).arrange(
            DOWN,
            buff=0.03,
        )

        text.move_to(
            box
        )

        return VGroup(
            box,
            text,
        )

    def gradient_badge(self, text):
        label = Text(
            text,
            font_size=13,
            weight="BOLD",
            color=self.GRADIENT,
        )

        background = RoundedRectangle(
            width=label.width + 0.16,
            height=label.height + 0.10,
            corner_radius=0.04,
            fill_color=self.BG,
            fill_opacity=0.93,
            stroke_width=0,
        )

        label.move_to(
            background
        )

        return VGroup(
            background,
            label,
        )

    def make_dense_gradient_badge(
        self,
        line_1,
        line_2,
    ):
        return VGroup(
            Text(
                line_1,
                font_size=12,
                color=self.GRADIENT,
            ),
            Text(
                line_2,
                font_size=12,
                color=self.GRADIENT,
            ),
        ).arrange(
            DOWN,
            buff=0.02,
        )

    def make_update_badge(self, text):
        label = Text(
            text,
            font_size=12,
            weight="BOLD",
            color=self.UPDATE,
        )

        border = RoundedRectangle(
            width=label.width + 0.18,
            height=label.height + 0.10,
            corner_radius=0.04,
            stroke_width=1.2,
            color=self.UPDATE,
        )

        label.move_to(
            border
        )

        return VGroup(
            border,
            label,
        )

    def make_training_strip(self):
        labels = [
            "Mini-batch",
            "Forward",
            "Loss",
            "Backprop",
            "Update",
        ]

        nodes = VGroup()

        for label in labels:
            box = RoundedRectangle(
                width=1.20,
                height=0.42,
                corner_radius=0.07,
                stroke_width=1.2,
                color=WHITE,
            )

            text = Text(
                label,
                font_size=13,
                weight="BOLD",
                color=self.TEXT,
            ).move_to(
                box
            )

            nodes.add(
                VGroup(
                    box,
                    text,
                )
            )

        nodes.arrange(
            RIGHT,
            buff=0.24,
        )

        arrows = VGroup()

        for left, right in zip(
            nodes[:-1],
            nodes[1:],
        ):
            arrows.add(
                Arrow(
                    left.get_right(),
                    right.get_left(),
                    buff=0.05,
                    stroke_width=1.8,
                    color=self.MUTED,
                    max_tip_length_to_length_ratio=0.18,
                )
            )

        # IMPORTANT:
        # No return/loop arrow.
        return VGroup(
            nodes,
            arrows,
        )

    def make_epoch_panel(self):
        label = Text(
            "30 epochs",
            font_size=13,
            weight="BOLD",
            color=self.TEXT,
        )

        note = Text(
            "reshuffle data each epoch",
            font_size=11,
            color=self.MUTED,
        )

        progress_bg = RoundedRectangle(
            width=4.6,
            height=0.12,
            corner_radius=0.04,
            stroke_width=1.0,
            color="#5A606B",
        )

        progress_fill = Rectangle(
            width=0.04,
            height=0.10,
            stroke_width=0,
            fill_opacity=1,
            color=self.UPDATE,
        )

        progress_fill.align_to(
            progress_bg,
            LEFT,
        )

        text_group = VGroup(
            label,
            note,
        ).arrange(
            RIGHT,
            buff=0.15,
        )

        text_group.next_to(
            progress_bg,
            UP,
            buff=0.07,
        )

        return Group(
            label,
            note,
            progress_bg,
            progress_fill,
        )

    def connect_right(
        self,
        left_group,
        right_group,
        color,
    ):
        return Arrow(
            left_group.get_right(),
            right_group.get_left(),
            buff=0.09,
            stroke_width=2.7,
            color=color,
            max_tip_length_to_length_ratio=0.16,
        )

    def connect_left(
        self,
        right_group,
        left_group,
        color,
        width,
    ):
        return Arrow(
            right_group.get_left(),
            left_group.get_right(),
            buff=0.10,
            stroke_width=width,
            color=color,
            max_tip_length_to_length_ratio=0.16,
        )

    # ================================================================
    # REAL IMAGE / FALLBACK
    # ================================================================

    def find_real_fer_image(self):
        script_path = Path(
            __file__
        ).resolve()

        repo_root = (
            script_path.parent.parent
        )

        final_directory = (
            repo_root
            / "data"
            / "final"
        )

        if not final_directory.exists():
            return None

        valid_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
        }

        for path in sorted(
            final_directory.rglob("*")
        ):
            if (
                path.is_file()
                and path.suffix.lower()
                in valid_extensions
            ):
                try:
                    return ImageMobject(
                        str(path)
                    )

                except Exception:
                    continue

        return None

    def make_pixel_grid(self):
        grid = VGroup()

        rows = 8
        cols = 8
        cell_size = 0.15

        pattern = [
            [0.15, 0.25, 0.40, 0.55, 0.55, 0.40, 0.25, 0.15],
            [0.20, 0.35, 0.60, 0.72, 0.72, 0.60, 0.35, 0.20],
            [0.25, 0.65, 0.15, 0.55, 0.55, 0.15, 0.65, 0.25],
            [0.30, 0.55, 0.45, 0.62, 0.62, 0.45, 0.55, 0.30],
            [0.25, 0.50, 0.55, 0.25, 0.25, 0.55, 0.50, 0.25],
            [0.20, 0.55, 0.35, 0.70, 0.70, 0.35, 0.55, 0.20],
            [0.18, 0.40, 0.62, 0.78, 0.78, 0.62, 0.40, 0.18],
            [0.15, 0.25, 0.38, 0.45, 0.45, 0.38, 0.25, 0.15],
        ]

        for row in range(rows):
            for col in range(cols):
                shade = pattern[row][col]

                square = Square(
                    side_length=cell_size,
                    stroke_width=0.20,
                )

                square.set_fill(
                    interpolate_color(
                        BLACK,
                        WHITE,
                        shade,
                    ),
                    opacity=1,
                )

                x = (
                    col
                    - (cols - 1) / 2
                ) * cell_size

                y = -(
                    row
                    - (rows - 1) / 2
                ) * cell_size

                square.move_to(
                    RIGHT * x
                    + UP * y
                )

                grid.add(
                    square
                )

        return grid


class NNArchitecturePresentation(ArchitectureBase):
    """
    Slower 16:9 presentation version.
    """
    NOTEBOOK_MODE = False


class NNArchitectureNotebook(ArchitectureBase):
    """
    Tighter version for GIF embedding in the notebook.
    """
    NOTEBOOK_MODE = True

