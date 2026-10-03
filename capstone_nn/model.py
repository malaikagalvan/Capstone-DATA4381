class Sequential:

    def __init__(
            self,
            *layers,
        ):
            self.layers = list(layers)


    def forward(
            self,
            x,
        ):
            for layer in self.layers:
                x = layer.forward(x)

            return x


    def backward(
            self,
            grad_output,
        ):
            for layer in reversed(
                    self.layers
                ):
                    grad_output = layer.backward(
                            grad_output
                        )

            return grad_output

    
    def update(
            self,
            learning_rate,
        ):
            for layer in self.layers:

                if hasattr(
                        layer,
                        "update",
                    ):
                        layer.update(
                                learning_rate
                            )
