import numpy as np
import os

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
    datasets = ['mnist', 'cifar']
    models = ['multinomial', 'gmm', 'lmm', 'sdp']
    samples = [500,5000,50000]
    component_params = [1,3,5,10,20]
    K = [1,2]
    lams = [0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0]
    ntrials = 10

    logprobs = np.zeros((len(datasets), len(models), len(samples), ntrials))
    rmse = np.zeros((len(datasets), len(models), len(samples), ntrials))
    for i,dataset in enumerate(datasets):
        for k,train_samples in enumerate(samples):
            for train_id in xrange(ntrials):
                for j,model in enumerate(models):
                    dargs = {'outputdir': 'experiments/pixels/results',
                             'train_samples': train_samples,
                             'model': model, 
                             'dataset': dataset,
                             'train_id': train_id}
                    cur_best = None
                    idx = (i,j,k,train_id)
                    if model == 'multinomial':
                        dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{train_samples}_{train_id}_score.csv'.format(**dargs))
                        get_score_if_best(dargs['outfile'], cur_best, logprobs, rmse, idx)
                    elif model in ('gmm', 'lmm'):
                        for num_components in component_params:
                            dargs['num_components'] = num_components
                            dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{train_samples}_{num_components}_{train_id}_score.csv'.format(**dargs))
                            cur_best = get_score_if_best(dargs['outfile'], cur_best, logprobs, rmse, idx)
                    elif model == 'sdp':
                        for order in K:
                            for lam in lams:
                                dargs['k'] = order
                                dargs['lam'] = lam
                                dargs['outfile'] = os.path.join(dargs['outputdir'], '{model}_{dataset}_{train_samples}_{k}_{lam}_{train_id}_score.csv'.format(**dargs))
                                cur_best = get_score_if_best(dargs['outfile'], cur_best, logprobs, rmse, idx)

            print ''
            print ''
            print 'Dataset: {dataset}-{train_samples}'.format(**dargs)
            for j,model in enumerate(models):
                print 'Model: {model}'.format(model=model)
                print 'Logprobs: {0:.2f} (Full: {1})'.format(logprobs[i,j,k].mean(), logprobs[i,j,k])
                print 'RMSE: {0:.2f} (Full: {1})'.format(rmse[i,j,k].mean(), rmse[i,j,k])
                print ''

    logprobs[logprobs == 0] = np.nan
    rmse[rmse == 0] = np.nan
    print '*** Summary of all results ***'.rjust(45)
    for i,dataset in enumerate(datasets):
        for k,train_samples in enumerate(samples):
            print '{dataset}-{train_samples}'.format(dataset=dataset, train_samples=train_samples).rjust(45)
            print 'log-prob'.rjust(35) + 'RMSE'.rjust(20)
            for j,model in enumerate(models):
                logprobs_str = '{0:.2f} ({1:.2f})'.format(np.nanmean(logprobs[i,j,k]), np.nanstd(logprobs[i,j,k])).rjust(20)
                rmse_str = '{0:.2f} ({1:.2f})'.format(np.nanmean(rmse[i,j,k]), np.nanstd(rmse[i,j,k])).rjust(20)
                print model.ljust(15) + logprobs_str + rmse_str
            print ''








