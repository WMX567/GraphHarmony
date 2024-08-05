folder_names = ['our']
pair_names = ['ir','ri']
dataset_dict = {'ir':['IMDB-BINARY', 'REDDIT-BINARY'],'ri':['REDDIT-BINARY','IMDB-BINARY']}
file_names = ['run_ours_ir.py']

for j, folder_name in enumerate(folder_names):
    for pair in pair_names:
        for i in range(5):
            file_name = folder_name + '/'+pair+str(i)+'_'+str(j)+'.sh'
            with open(file_name, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('#SBATCH --partition=gpu\n')
                f.write('#SBATCH --gres=gpu:p100:1\n')
                f.write('#SBATCH --nodes=1\n')
                f.write('#SBATCH --ntasks=1\n')
                f.write('#SBATCH --cpus-per-task=4\n')
                f.write('#SBATCH --mem=32GB\n')
                f.write('#SBATCH --time=3:00:00\n')
                f.write('#SBATCH --output='+pair+str(i)+'_'+str(j)+'.out'+'\n')
                f.write('\n\n')
                f.write('module purge\n')
                f.write('eval "$(conda shell.bash hook)"\n')
                f.write('conda activate pyCLGL\n')
                f.write('\n\n')
                f.write('python /scratch1/mengxiwu/GraphHarmony/'+file_names[j]+
                ' --backbone gat --r '+str(i)+' --data_path /scratch1/mengxiwu/GraphCLS/'+ 
                ' --src_data '+ dataset_dict[pair][0] + ' --tar_data '+ dataset_dict[pair][1]
                + ' --device cuda\n')
            f.close()

for j, folder_name in enumerate(folder_names):
    with open(folder_name+'.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --partition=gpu\n')
        f.write('#SBATCH --gres=gpu:1\n')
        f.write('#SBATCH --nodes=1\n')
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --cpus-per-task=4\n')
        f.write('#SBATCH --mem=32GB\n')
        f.write('#SBATCH --time=00:05:00\n')
        f.write('#SBATCH --output='+pair+str(i)+'_'+str(j)+'.out'+'\n')
        f.write('\n\n')
        f.write('module purge\n')
        f.write('eval "$(conda shell.bash hook)"\n')
        f.write('conda activate py39\n')
        f.write('\n\n')
        for pair in pair_names:
            for i in range(5):
                file_name = folder_name + '/'+pair+str(i)+'_'+str(j)+'.sh'
                f.write('sbatch '+file_name+'\n')

folder_names = ['OurBase']
with open('test.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --partition=gpu\n')
    f.write('#SBATCH --gres=gpu:1\n')
    f.write('#SBATCH --nodes=1\n')
    f.write('#SBATCH --ntasks=1\n')
    f.write('#SBATCH --cpus-per-task=4\n')
    f.write('#SBATCH --mem=32GB\n')
    f.write('#SBATCH --time=00:20:00\n')
    f.write('#SBATCH --output=test.out'+'\n')
    f.write('\n\n')
    f.write('module purge\n')
    f.write('eval "$(conda shell.bash hook)"\n')
    f.write('conda activate py39\n')
    f.write('\n\n')
    for j, folder_name in enumerate(folder_names):
        for pair in pair_names:
            f.write('python /scratch1/mengxiwu/GraphHarmony/test_ir.py'+
            ' --backbone gat -m '+folder_name+' --data_path /scratch1/mengxiwu/GraphCLS/'+ 
            ' --src_data '+ dataset_dict[pair][0] + ' --tar_data '+ dataset_dict[pair][1]
            + ' --device cuda\n')
f.close()
