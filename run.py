import DNN3

datano=11
DNN3.RunMLP(DataNo=datano, SimNo_start=sim, SimNo_end=sim+3, ModelNo=1, Nloci=20, Nunit=20, Naddlayer=0, Rdrop=0.0, Sbatch=32, Nepoch=1000, Vsplit=0.2, AddBN=True, Bias=True)


