import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from utils import read_data_sets

if __name__ == '__main__':
    models = ['multinomial', 'gmm', 'lmm', 'multiscale', 'trendfiltering', 'trendfiltering-multiscale']
    model_names = ['Multinomial', 'GMM', 'LMM', 'Unsmoothed DP', 'Smoothed MN', 'SDP']
    dist_types = ['edge_biased']
    sample_sizes = [500,1000,3000,5000,10000,15000,30000,60000]

    test_idx = 3
    dargs = {}
    dargs['train_id'] = 0
    for dist in dist_types:
        mnist = read_data_sets('experiments/mnist/data', dist_type=dist, ignore_bins_in_filename=True)
        dargs['dist_type'] = dist
        dargs['dist_name'] = dist.replace('_','-')
        print ''
        print dist
        for j,N in enumerate(sample_sizes):
            dargs['train_samples'] = N
            print 'N={train_samples}'.format(**dargs)
            with sns.axes_style('white'):
                plt.rc('font', weight='bold')
                plt.rc('grid', lw=2)
                plt.rc('lines', lw=3)
                colors = ['gray', 'blue', 'orange', 'green', 'purple']
                styles = ['solid', 'dashed', 'dashdot', 'solid', 'solid']
                for i,(model,name) in enumerate(zip(models,model_names)):
                    dargs['model'] = model
                    filename = 'experiments/mnist/results/{model}_{dist_type}_{train_samples}_{train_id}_fits.csv'.format(**dargs)
                    fits = np.loadtxt(filename, delimiter=',')
                    plt.figure(1)
                    plt.plot(np.arange(fits.shape[1]), mnist.densities[mnist.test.labels[test_idx]].weights, c='gray', ls='solid', lw=4, label='Truth', alpha=0.5)
                    plt.plot(np.arange(fits.shape[1]), fits[test_idx], c='orange', ls='solid',  lw=4, label=name)
                    plt.ylabel('Probability', weight='bold', fontsize=24)
                    xticks = list(plt.gca().get_xticks())
                    while (fits.shape[1]-1) / float(xticks[-1]) < 1.1:
                        xticks = xticks[:-1]
                    while xticks[0] < 0:
                        xticks = xticks[1:]
                    xticks.append(fits.shape[1]-1)
                    xticks = list(sorted(xticks))
                    plt.gca().set_xticks(xticks)
                    plt.xlim([-5,fits.shape[1]+5])
                    plt.legend(loc='upper right')
                    plt.savefig('plots/mnist-fits-{dist_name}-{model}-{train_samples}.pdf'.format(**dargs), bbox_inches='tight')
                    plt.clf()
                    plt.close()


            
            

            
            


