folder_names = ['our']
pair_names = ['ot','ow','od','to','tw','td','wo','wt','wd','do','dt','dw']
dataset_dict = {'do':['digg', 'oag'],'dt':['digg','twitter'],
'dw':['digg', 'weibo'],'od':['oag','digg'],'ot':['oag','twitter'],
'ow':['oag','weibo'],'td':['twitter','digg'],'to':['twitter','oag'],
'tw':['twitter','weibo'],'wd':['weibo','digg'],
'wo':['weibo','oag'],'wt':['weibo','twitter']}
file_names = ['run_ours.py']

# for j, folder_name in enumerate(folder_names):
#     for pair in pair_names:
#         for i in range(5):
#             file_name = folder_name + '/'+pair+str(i)+'_'+str(j)+'.sh'
#             with open(file_name, 'w') as f:
#                 f.write('#!/bin/bash\n')
#                 f.write('#SBATCH --partition=gpu\n')
#                 f.write('#SBATCH --gres=gpu:p100:1\n')
#                 f.write('#SBATCH --nodes=1\n')
#                 f.write('#SBATCH --ntasks=1\n')
#                 f.write('#SBATCH --cpus-per-task=4\n')
#                 f.write('#SBATCH --mem=32GB\n')
#                 f.write('#SBATCH --time=2:00:00\n')
#                 f.write('#SBATCH --output='+pair+str(i)+'_'+str(j)+'.out'+'\n')
#                 f.write('\n\n\n\n')
#                 f.write('module purge\n')
#                 f.write('eval "$(conda shell.bash hook)"\n')
#                 f.write('conda activate pyCLGL\n')
#                 f.write('\n\n\n\n')

#                 f.write('python /scratch1/mengxiwu/GraphHarmony/'+file_names[j]+
#                 ' --backbone gat --r '+str(i)+' --data_path /scratch1/mengxiwu/GraphHarmony/infodata/data/'+ 
#                 ' --src_data '+ dataset_dict[pair][0] + ' --tar_data '+ dataset_dict[pair][1]
#                 + ' --device cuda\n')

# f.close()

foldername = folder_names[0]+".sh"
with open(foldername, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --partition=main\n')
    f.write('#SBATCH --nodes=1\n')
    f.write('#SBATCH --ntasks=1\n')
    f.write('#SBATCH --cpus-per-task=4\n')
    f.write('#SBATCH --time=00:05:00\n')
    f.write('#SBATCH --output=our.out'+'\n')
    f.write('\n\n\n\n')

    for j, folder_name in enumerate(folder_names):
        for pair in pair_names:
            for i in range(5):
                f.write(foldername+'/'+pair+str(i)+'_'+str(j)+'.sh\n')
f.close()
