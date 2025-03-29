from combinetf2.physicsmodels.ratio import Ratio


class Normratio(Ratio):
    """
    Same as Ratio but the numerator and denominator are normalized
    """

    name = "normratio"
    normalize = True

    def init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, values):
        num = self.num.select(values, normalize=True, inclusive=True)
        den = self.den.select(values, normalize=True, inclusive=True)

        return num / den

    def compute_per_process(self, values):
        num = self.num.select(values, normalize=True, inclusive=False)
        den = self.den.select(values, normalize=True, inclusive=False)

        return num / den
