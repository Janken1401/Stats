import matplotlib.pyplot as plt
from stats import Stats, JointStats
from scipy.stats import kurtosis, skew
import numpy as np


LW = 2


def line():
    print("=====================================")


def v_space():
    print("\n")


class AFFICHAGE:
    def __init__(self, TP='TP1_2', question='all', param='default'):
        if param == 'default':
            self.param = dict(N=10000,
                              npdf=250)
        else:
            self.param = param
        self.TP = TP
        self.question = question
        self.run()

    def run(self):
        if self.TP == 'TP1_2':
            r = TP1_2(self.question, self.param)
            r.run()
        if self.TP == 'TP3':
            r = TP3(self.question, self.param)
            r.run()
        if self.TP == 'TP4':
            r = TP4(self.question, self.param)
            r.run()


class TP1_2:
    def __init__(self, question, params):
        self.question = question
        self.npdf = params['npdf']
        self.N = params['N']
        self.X = np.random.normal(5, 0.5, self.N)
        self.r = Stats(self.X, self.npdf)

    def run(self):
        if self.question == 'all':
            self.question1()
            self.question2()
            self.question3()
            self.question4()
            self.question5()
            self.question6()
            self.question7()
        else :
            for i in self.question:
                if i == '1':
                    self.question1()
                if i == '2':
                    self.question2()
                if i == '3':
                    self.question3()
                if i == '4':
                    self.question4()
                if i == '5':
                    self.question5()
                if i == '6':
                    self.question6()
                if i == '7':
                    self.question7()

    def question1(self):
        v_space()
        line()
        print('question 1 :')
        line()
        print('pdf : \n', self.r.pdf)
        line()
        print('xlevels :\n', self.r.levels)
        line()
        v_space()

    def question2(self):
        plt.figure()
        plt.title("Representation of mypdf for N = %d and npdf = %d" %
                  (self.N, self.npdf))
        plt.plot(self.r.levels, self.r.pdf, label='mypdf')
        plt.hist(self.X, bins=self.npdf, density=True, label='Hist')
        plt.legend()
        plt.show()

    def question3(self):
        v_space()
        line()
        print('question 3 :')
        line()
        print('intégrate of my_pdf = %f' % (self.r.dx*np.sum(self.r.pdf)))
        line()
        v_space()

    def question4(self):
        v_space()
        line()
        print('question 4:')
        line()
        print('mean = %f' % self.r.mean())
        print('variance = %f' % self.r.variance())
        print('coefficient of skewness = %f' % self.r.skewness())
        print('coefficient of kurtosis = %f' % self.r.kurtosis())
        line()
        v_space()

    def question5(self):
        v_space()
        line()
        print('question 5 :')
        line()
        print('mean = %f' % np.mean(self.X))
        print('variance = %f' % np.var(self.X))
        print('coefficient of skewness = %f' % skew(self.X))
        print('coefficient of kurtosis = %f' % kurtosis(self.X, fisher=False))
        line()
        v_space()

    def question6(self):
        plt.figure()
        plt.title("Representation of mycdf for N = %d and npdf = %d" %
                  (self.N, self.npdf))
        plt.plot(self.r.levels, self.r.cdf())
        plt.show()

    def question7(self):
        plt.figure()
        plt.title("Representation of the derivation of mycdf for N = %d and npdf = %d" %
                  (self.N, self.npdf))
        df_dx = self.r.refind_pdf()
        plt.plot(self.r.levels, df_dx, label='refind pdf')
        plt.hist(self.X, bins=self.npdf, density=True, label='Hist')
        plt.legend()
        plt.show()


class TP3:
    def __init__(self, question, params):
        self.question = question
        self.npdf = params['npdf']
        self.N = params['N']
        self.X = params['X']
        self.Y = params['Y']
        self.r = JointStats(self.X, self.Y, self.npdf)

    def run(self):
        if self.question == 'all':
            self.question1()
            self.question2()
            self.question4()
            self.question5()
        else :
            for i in self.question:
                if i == '1':
                    self.question1()
                if i == '2':
                    self.question2()
                if i == '4':
                    self.question4()
                if i == '5':
                    self.question5()

    def question1(self):
        v_space()
        line()
        print('question 1 :')
        line()
        print('pdf : \n', self.r.pdf)
        line()
        print('xlevels :\n', self.r.stats_x.levels)
        line()
        print('ylevels :\n', self.r.stats_y.levels)
        line()
        v_space()
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.title("Representation of mypdf for X with N = %d and npdf = %d" %
                  (self.N, self.npdf))
        plt.plot(self.r.stats_x.levels, self.r.stats_x.pdf, label='mypdf')
        plt.hist(self.X, bins=self.npdf, density=True, label='Hist')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title("Representation of mypdf for y with N = %d and npdf = %d" %
                  (self.N, self.npdf))
        plt.plot(self.r.stats_y.levels, self.r.stats_y.pdf, label='mypdf')
        plt.hist(self.Y, bins=self.npdf, density=True, label='Hist')
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        x, y = np.meshgrid(self.r.stats_x.levels, self.r.stats_y.levels)
        plt.title("Representation of mypdf for X and Y with N = %d and npdf = %d" %
                  (self.N, self.npdf))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, self.r.pdf, antialiased=False, cmap='inferno')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def question2(self):
        v_space()
        line()
        print('question 2 :')
        line()
        print('correlation : \n', self.r.correlation())
        line()
        v_space()

    def question4(self):
        v_space()
        line()
        print('question 4:')
        line()
        print('mean = %f' % self.r.mean())
        print('variance = %f' % self.r.variance())
        print('coefficient of skewness = %f' % self.r.skewness())
        print('coefficient of kurtosis = %f' % self.r.kurtosis())
        line()
        v_space()

    def question5(self):
        v_space()
        line()
        print('question 5 :')
        line()
        print('mean = %f' % np.mean(self.X))
        print('variance = %f' % np.var(self.X))
        print('coefficient of skewness = %f' % skew(self.X))
        print('coefficient of kurtosis = %f' % kurtosis(self.X, fisher=False))
        line()
        v_space()

# class TP4:
#     def __init__(self, question):
#         self.ddd = dd
