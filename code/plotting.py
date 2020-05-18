import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_time_depend(plot_dict, n, Fun, problem_flag = 1, task_name = None):
    with plt.style.context('seaborn-paper'):
        f = plt.figure(figsize=(10,10))
        for opt in plot_dict.keys():
            if plot_dict[opt][1] is not None:
                plt.loglog(plot_dict[opt][1],  
                        [np.linalg.norm(Fun(x), 2) for x in plot_dict[opt][0]] , 
                        '--o', label=opt, linewidth=7)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.legend(fontsize=30)
        plt.title('', fontsize=30)
        plt.xlabel('Time, ms', size=30)
        plt.ylabel(r"$||\nabla f(x)||$", size=30)
        plt.grid()
    plt.show()
    #Здесь бы добавить окончательно проверку на None и подстановку произвольной задачи
    name_task = "roze" if problem_flag == 1 else "trig" if problem_flag == 2 else "noname" 
    f.savefig(f"Results/time_n{n}_{name_task}.pdf", bbox_inches='tight')

def plot_iter_depend_norm(plot_dict, dim_n, F, problem_flag = 1, task_name = None):
    with plt.style.context('seaborn-paper'):
        f = plt.figure(figsize=(10,10))
        for opt in plot_dict.keys():
            plt.loglog(np.arange(len(plot_dict[opt][0])),  
                    [np.linalg.norm(F(x), 2) for x in plot_dict[opt][0]] , 
                    '--o', label=opt, linewidth=7)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.legend(fontsize=30)
        plt.title('', fontsize=30)
        plt.xlabel('Iterations', size=30)
        plt.ylabel(r"$||f(x)||$", size=30)
        plt.grid()
    plt.show()
    name_task = "roze" if problem_flag == 1 else "trig" if problem_flag == 2 else "noname"
    f.savefig(f"Results/iter_n{dim_n}_{name_task}_norm.pdf", bbox_inches='tight')
    
def plot_iter_depend_delta(plot_dict, x_true, n, problem_flag = 1, task_name = None):
    with plt.style.context('seaborn-paper'):
        f = plt.figure(figsize=(10,10))
        for opt in plot_dict.keys():
            plt.loglog(np.arange(len(plot_dict[opt][0])),  
                    [np.linalg.norm(x - x_true, 2) for x in plot_dict[opt][0]] , 
                    '--o', label=opt, linewidth=7)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.legend(fontsize=30)
        plt.title('', fontsize=30)
        plt.xlabel('Iterations', size=30)
        plt.ylabel(r"$||\Delta x_k||$", size=30)
        plt.grid()
    plt.show()
    name_task = "roze" if problem_flag == 1 else "trig" if problem_flag == 2 else "noname"
    f.savefig(f"Results/iter_n{n}_{name_task}_delta.pdf", bbox_inches='tight')

def plot_iter_depend_grad(plot_dict, n, F, problem_flag = 1, task_name = None):
    with plt.style.context('seaborn-paper'):
        f = plt.figure(figsize=(10,10))
        for opt in plot_dict.keys():
            plt.loglog(np.arange(len(plot_dict[opt][0])),  
                    [np.linalg.norm(F(x), 2) for x in plot_dict[opt][0]] , 
                    '--o', label=opt, linewidth=7)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.legend(fontsize=30)
        plt.title('', fontsize=30)
        plt.xlabel('Iterations', size=30)
        plt.ylabel(r"$||\nabla f(x)||$", size=30)
        plt.grid()
    plt.show()
    name_task = "roze" if problem_flag == 1 else "trig" if problem_flag == 2 else "noname"
    f.savefig(f"Results/iter_n{n}_{name_task}_grad.pdf", bbox_inches='tight')
