import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import os
from pygfl.utils import pretty_str

if __name__ == '__main__':
    models = ['multinomial', 'gmm', 'lmm', 'multiscale', 'trendfiltering', 'trendfiltering-multiscale']
    model_names = ['Multinomial', 'GMM', 'LMM', 'Unsmoothed DP', 'Smoothed MN', 'SDP']
    dist_types = ['gmm', 'edge_biased']
    sample_sizes = [500, 1000, 3000, 5000, 10000, 15000, 30000, 60000]
    ntrials = 10

    dargs = {}
    for dist in dist_types:
        scores = np.zeros((len(models), len(sample_sizes), ntrials))
        dargs['dist_type'] = dist
        print ''
        print dist
        for i,model_name in enumerate(models):
            dargs['model'] = model_name
            for j,N in enumerate(sample_sizes):
                dargs['train_samples'] = N
                for trial in xrange(ntrials):
                    dargs['train_id'] = trial
                    filename = 'experiments/mnist/results/{model}_{dist_type}_{train_samples}_{train_id}_score.csv'.format(**dargs)
                    if os.path.exists(filename):
                        arr = np.loadtxt(filename, delimiter=',')
                        if len(arr) >= 2:
                            scores[i,j,trial] = arr[1]
                        else:
                            scores[i,j,trial] = np.nan
                    else:
                        scores[i,j,trial] = np.nan
            scores[scores == 0] = np.nan
            print '\t' + model_name
            print '\t' + pretty_str(scores[i])

        scores = np.nanmean(scores,axis=2)

        with sns.axes_style('white'):
            plt.rc('font', weight='bold')
            plt.rc('grid', lw=2)
            plt.rc('lines', lw=3)
            
            colors = ['gray', 'blue', 'brown', 'orange', 'green', 'purple', 'goldenrod']
            styles = ['solid', 'dashed', 'dashed', 'dashdot', 'dashdot', 'solid', 'solid']

            plt.figure(1)
            for i, (score, c, ls, name) in enumerate(zip(scores, colors, styles, model_names)):
                plt.plot(np.log(sample_sizes), score, c=c, ls=ls,  lw=4, label=name)
            plt.ylabel('TV Error', weight='bold', fontsize=24)
            plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
            plt.legend(loc='upper right', ncol=2)
            plt.savefig('plots/mnist-scores-{0}.pdf'.format(dist.replace('_','-')), bbox_inches='tight')
            plt.clf()
            plt.close()



