import numpy as np
import os
from tfsdp.utils import pretty_str
from utils import dataset_details

def get_score_if_best(filename, cur_best, logprobs, rmse, idx):
    if not os.path.exists(filename):
        return cur_best
    results = np.loadtxt(dargs['outfile'])
    validation_loss = results[0]
    cur_logprobs = results[1]
    cur_rmse = results[2]
    if np.isnan(validation_loss) or np.isnan(cur_logprobs) or np.isnan(cur_rmse):
        return cur_best
    if cur_best is None or validation_loss < cur_best:
        logprobs[idx] = cur_logprobs
        rmse[idx] = cur_rmse
        return validation_loss
    return cur_best

if __name__ == '__main__':
    datasets = ['auto_mpg', 'housing', 'abalone', 
                'energy_efficiency', 'parkinsons',
                'student_performance', 'concrete']
    models = ['multinomial', 'gmm', 'lmm', 'sdp']
    component_params = [1,3,5,10,20]
    K = [1,2]
    lams = [0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0]
    ntrials = 10

    logprobs = np.zeros((len(datasets), 4, ntrials))
    rmse = np.zeros((len(datasets), 4, ntrials))
    for i,dataset in enumerate(datasets):
        for train_id in xrange(ntrials):
            for j,model in enumerate(models):
                dargs = {'outputdir': 'experiments/uci/results',
                         'model': model, 
                         'dataset': dataset,
                         'train_id': train_id}
                cur_best = None
                idx = (i,j,train_id)
                if model == 'multinomial':
                    dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{train_id}_score.csv'.format(**dargs))
                    get_score_if_best(dargs['outfile'], cur_best, logprobs, rmse, idx)
                elif model in ('gmm', 'lmm'):
                    for num_components in component_params:
                        dargs['num_components'] = num_components
                        dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{num_components}_{train_id}_score.csv'.format(**dargs))
                        cur_best = get_score_if_best(dargs['outfile'], cur_best, logprobs, rmse, idx)
                elif model == 'sdp':
                    for k in K:
                        for lam in lams:
                            dargs['k'] = k
                            dargs['lam'] = lam
                            dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{k}_{lam}_{train_id}_score.csv'.format(**dargs))
                            cur_best = get_score_if_best(dargs['outfile'], cur_best, logprobs, rmse, idx)

        print ''
        print ''
        print 'Dataset: {dataset}'.format(**dargs)
        for j,model in enumerate(models):
            print 'Model: {model}'.format(model=model)
            print 'Logprobs: {0:.2f} (Full: {1})'.format(logprobs[i,j].mean(), pretty_str(logprobs[i,j], ignore=0))
            print 'RMSE: {0:.2f} (Full: {1})'.format(rmse[i,j].mean(), pretty_str(rmse[i,j], ignore=0))
            print ''

    print '*** Summary of all results ***'.rjust(25)
    for i,dataset in enumerate(datasets):
        nfeatures, nbins, nsamples = dataset_details(dataset, 'experiments/uci/data')
        print ''
        print '{dataset}\nFeatures: {nfeatures} Samples: {nsamples} CPE shape: {nbins}'.format(dataset=dataset, nfeatures=nfeatures,
                                                                                                nsamples=nsamples, nbins=nbins)
        print 'log-prob'.rjust(35) + 'RMSE'.rjust(20)
        for j,model in enumerate(models):
            logprobs_str = '{0:.2f} ({1:.2f})'.format(logprobs[i,j].mean(), logprobs[i,j].std()).rjust(20)
            rmse_str = '{0:.2f} ({1:.2f})'.format(rmse[i,j].mean(), rmse[i,j].std()).rjust(20)
            print model.ljust(15) + logprobs_str + rmse_str
        print ''








