import numpy as np

class Flatten:
    '''
    Flatten all non-batch dimensions into one usable dimension

    Example:
        (N,48,48) -> (N, 2304)

    '''

    def __init__(self):
        self.input_shape=None

    def forward(self, x):
        '''
        Flatten input while preserving batch dimension
        '''

        self.input_shape = x.shape

        return x.reshape(
                x.shape[0],
                -1,
            ) # the -1 is not an indexing step it is a numpy feature that 
        # in this case will reshape our dimensions to vectors
        # IMPORTANT! reshape() does not modify x, it returns another array object

    def backward(self, grad_output):
        '''
        Restore the gradient to the original input shape
        '''

        if self.input_shape is None:
            raise RuntimeError(
                    "forward() must be called before backward()."
                )

        return grad_output.reshape(
                self.input_shape
            )


class Dense:

    def __init__(
            self,
            input_size,
            output_size,
        ):

            self.input_size = input_size
            self.output_size = output_size

            self.weights = (
                    np.random.randn(
                        input_size,
                        output_size,
                    )
                    * np.sqrt(2.0 / input_size)
                )

            self.bias = np.zeros(
                    output_size,
                    dtype=np.float32,
                )

            self.input = None

            self.grad_weights = None
            self.grad_bias = None


    def forward(self, x):
        '''
        Compute the output of the dense layer

        x:
            Shape(batch_size, input_size)

        returns:
            Shape(batch_size, output_size)

        '''

        self.input = x

        return (
            x @ self.weights + self.bias
            )



