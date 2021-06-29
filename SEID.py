SEIR_PARAM = namedtuple('SEIRparm', ['alpha1', 'alpha2', 'beta', 'sigma', 'gamma'])

class SEIR(object):
    def __init__(self, P=None):
        self.P = P
    
    def _forward(self, S, E, I, D, C, param, max_iter):
        a1, a2, s, g, b = param 
        est = dp.Table(columns=['S', 'E', 'I', 'D', 'C'])
        for t in range(max_iter):
            S_ = S - a1 * S * E / N  - a2 * S * I / N
            E_ = E + a1 * S * E / N + a2 * S * I / N - b * E
            I_ = I + b * E - s * I - g * I
            D_ = D + g * I
            C_ = C + s * I
            S, E, I, D, C = S_, E_, I_, D_, C_
            est.append_row([S, E, I, D, C])
        return est
    
    def _loss(self, obs, est):
        assert len(obs) == len(est)
        loss = ((np.log2(obs + 1) - np.log2(est + 1)) ** 2).sum()
        self.lossing.append(loss)
        return loss
    
    def _optimize(self, param, s, e, i, d, c, obs):
        est = self._forward(s, e, i, d, c, param, len(obs))
        return self._loss(obs, est['I', 'D', 'C'].toarray())
    
    def fit(self, initS, initE, initI, initD, initC, Y):
        self.lossing = []
        param = [(0, 1),] * 5
        lw = [0, 0, 0, 0, 0.0] * 5 # lower bound
        up = [1, 1, 1, 1, 1] # upper bound
        args = (initS, initE, initI, initD, initC, Y['case', 'dead', 'recover'].toarray())
        result = dual_annealing(self._optimize, bounds=list(zip(lw, up)), args=args, seed=9527, maxiter=100)['x']
        minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
        result = differential_evolution(self._optimize, 
                                        bounds = list(zip(lw, up)), 
                                        args = args, 
                                        strategy='best1bin', 
                                        maxiter=1000, 
                                        popsize=15)['x']
        self.P = SEIR_PARAM(*result)
    
    def score(self, initS, initE, initI, initD, initC, Y, plot=False):
        est = self.predict(initS, initE, initI, initD, initC, len(Y))['I', 'D', 'C']
        loss = self._loss(Y['case', 'dead', 'recover'].toarray(), est.toarray())
        est.columns = ['case', 'dead', 'recover']
        if plot:
            self.plot_predict(Y, est)
            print(' - avg incubation period：%.2f day' % (1.0 / self.P.beta))
            print(' - R0：%.2f' % (self.P.alpha1 / self.P.beta + (self.P.alpha2 / self.P.sigma + self.P.alpha2 / self.P.gamma)/ 2))
            print(' - loss：%.4f' % loss)
        return loss
    
    def plot_error(self):
        plt.plot(self.lossing, label=u'accuracy')
        plt.legend()
        plt.show()
    
    def plot_predict(self, obs, est):
        print(type(obs),type(est))
        for label, color in zip(obs.keys(), ['red', 'black', 'green']):
            plt.plot(obs[label], color=color)
            plt.plot(est[label], color='m')
            plt.legend()
            plt.show()
            
    def predict(self, initS, initE, initI, initD, initC, T):
        return self._forward(initS, initE, initI, initD, initC, self.P, T)
    
    def storeParameters(self):
        print(self.P)
        return self.P
    
    def searchBestParam(seir):
    min_loss, best_param, likeli_potential = float('inf'), None, 0
    for potential in range(200, 1000, 100):
        seir.fit(S, potential, I, D, C, train)
        loss = seir.score(S, potential, I, D, C, train)
        if loss < min_loss:
            print('E：%.4f | loss： %.6f' % (potential, loss))
            min_loss, best_param, likeli_potential = loss, seir.P, potential
    seir.P = best_param
    seir.score(S, likeli_potential, I, D, C, Y=train, plot=True)
    return seir, likeli_potential

findPara = SEIR()
seir, potentials = searchBestParam(findPara)
findPara.storeParameters()
