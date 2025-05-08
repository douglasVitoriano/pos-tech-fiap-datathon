class DataPreparation:

    def merge(self, left_on, right_on, how='inner'):
        return self.df1.merge(self.df2, left_on=left_on, right_on=right_on, how=how)