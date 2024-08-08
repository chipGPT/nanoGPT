class Measure:
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class AbsMax(Measure):
    def __call__(self, input, keepdim):
        return input.abs().max(dim=-1, keepdim=keepdim).values if keepdim else input.abs().max()

class AbsMean(Measure):
    def __call__(self, input, keepdim):
        return input.abs().mean(dim=-1, keepdim=keepdim) if keepdim else input.abs().mean()

class AbsMedian(Measure):
    def __call__(self, input, keepdim):
        return input.abs().median(dim=-1, keepdim=keepdim).values if keepdim else input.abs().median()
