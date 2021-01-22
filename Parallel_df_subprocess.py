'''
Parallel application of subprocess calls by parallelising use of pandas dfs
'''

%%time
from nilearn import datasets, image
niimg = datasets.load_mni152_template()
nodes_temp = nodes.iloc[:400,:].copy()
def parallelize_dataframe(df, func, n_cores=20):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
def add_features(nodes_temp):
    nodes_temp['mni_x'], nodes_temp['mni_y'], nodes_temp['mni_z'] = image.coord_transform(nodes_temp['x'], nodes_temp['y'], nodes_temp['z'], niimg.affine)
    bashCommand = 'atlasquery -a "Talairach Daemon Labels" -c '+str(np.array(nodes_temp['mni_x'])[0])+','+str(np.array(nodes_temp['mni_y'])[0])+','+str(np.array(nodes_temp['mni_z'])[0]) ##prepare the subprocess for FSL atlasquery
    region = subprocess.check_output(['bash','-c',bashCommand])
    
    nodes_temp['label0']=region.decode().split('.')[0].rstrip("\n") ##execute it
    nodes_temp['label1']=region.decode().split('.')[1].rstrip("\n") ##execute it
    nodes_temp['label2']=region.decode().split('.')[2].rstrip("\n") ##execute it
    nodes_temp['label3']=region.decode().split('.')[3].rstrip("\n") ##execute it
    nodes_temp['label4']=region.decode().split('.')[4].rstrip("\n") ##execute it
    
    return nodes_temp
nodes_temp = parallelize_dataframe(nodes_temp, add_features)
