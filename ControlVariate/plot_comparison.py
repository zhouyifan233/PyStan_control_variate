import matplotlib.pylab as plt
import os


def plot_comparison(old_mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples, fig_path='savefig', fig_name='figure.png', fig_size=(8, 8)):
    num_samples = old_mcmc_samples.shape[0]
    dim = old_mcmc_samples.shape[1]


    fig, ax = plt.subplots(dim, figsize=fig_size)
    for i in range(dim):
        ax[i].plot(range(0, num_samples), old_mcmc_samples[:, i], color='blue',
                      label='original mcmc samples')
        ax[i].plot(range(0, num_samples), cv_linear_mcmc_samples[:, i], color='red',
                      label='with linear control variate')
        ax[i].plot(range(0, num_samples), cv_quad_mcmc_samples[:, i], color='black',
                      label='with quadratic control variate')
        ax[i].legend(loc=1, fancybox=True, bbox_to_anchor=(1.0, 1.5))
        ax[i].set_xlabel('mcmc iteration')
        ax[i].set_ylabel('Parameter')
        ax[i].set_title('Parameter ' + str(i+1), loc='left')
    fig.subplots_adjust(hspace=1.0)
    fig.savefig(os.path.join(fig_path, fig_name))


