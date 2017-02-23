import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import os
from pygfl.utils import pretty_str

if __name__ == '__main__':
    models = ['trendfiltering-multiscale']
    model_names = ['SDP']
    dist_types = ['edge_biased', 'gmm']
    dist_names = ['Edge-Biased', 'GMM']
    sample_sizes = [500, 1000, 3000, 5000, 10000, 15000, 30000, 60000]
    ntrials = 10

    dargs = {}
    for model, name in zip(models, model_names):
        dargs['model'] = model
        scores = np.zeros((len(dist_types), len(sample_sizes), ntrials, 2))
        for i,dist in enumerate(dist_types):
            dargs['dist_type'] = dist
            print ''
            print dist
            for j,N in enumerate(sample_sizes):
                dargs['train_samples'] = N
                for trial in xrange(ntrials):
                    dargs['train_id'] = trial
                    filename = 'experiments/mnist/results/{model}_{dist_type}_{train_samples}_{train_id}_score.csv'.format(**dargs)
                    scores[i,j,trial] = (np.loadtxt(filename, delimiter=',')[3:5]) if os.path.exists(filename) else np.nan
            scores[scores == 0] = np.nan
            print '\t' + name
            print '\t' + pretty_str(scores[i,j,:,0], decimal_places=6)
            print '\t' + pretty_str(scores[i,j,:,1], decimal_places=6)

        scores = np.nanmean(scores,axis=2)

        with sns.axes_style('white'):
            plt.rc('font', weight='bold')
            plt.rc('grid', lw=2)
            plt.rc('lines', lw=3)

            plt.figure(1)
            colors = ['blue', 'orange']
            for dname, dscores, c in zip(dist_names, scores, colors):
                lns1 = plt.plot(np.log(sample_sizes), np.log(dscores[:,1]), c=c, ls='solid', marker='s', lw=4, label=dname)
            plt.ylim([-9,-3])
            plt.ylabel('Log(Lambda)', weight='bold', fontsize=24)
            plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
            # Plot everything on the same legend
            plt.legend(loc='upper right')
            plt.savefig('plots/mnist-lambdas.pdf', bbox_inches='tight')
            plt.clf()
            plt.close()



