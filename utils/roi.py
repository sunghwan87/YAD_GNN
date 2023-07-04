import pandas as pd

def convert_roi_name_to_disp(roi_df: pd.DataFrame) -> pd.DataFrame:
    roi_disp_order = pd.CategoricalDtype(['Default', 'Control', 'Limbic', 'Sal/VentAttn', 'DorsAttn','TempPar', 'SomMot', 'Visual', 'Subcortical'], ordered=True)
    roi_disp_mapper = dict({'TempPar': 'TempPar',
                            'Default':'Default',
                            'Cont':'Control',
                            'Limbic':'Limbic',
                            'SalVent':'Sal/VentAttn',
                            'Dors':'DorsAttn',
                            'SomMot':'SomMot',
                            'Vis':'Visual',
                            'Subcortical':'Subcortical'})

    for kw in roi_disp_mapper.keys():
        kw_match = list(filter(lambda x: kw in x, roi_df['label']))
        mask = roi_df['label'].apply(lambda x: x in kw_match).values
        roi_df.loc[mask,'category'] = kw
    roi_df['disp'] = roi_df['category'].map(roi_disp_mapper).astype(roi_disp_order)
    roi_df.sort_values(['disp'], inplace=True)
    return roi_df.set_index('label')

def load_roi_labels(file_path: str) -> pd.DataFrame:
    roi_names_df = pd.read_csv(file_path, delimiter=r'\s+')
    roi_names = roi_names_df['NAME'].to_list()
    roi_names = ['_'.join(name.split('_')[3:] + name.split('_')[0:1]) for name in roi_names]
    roi_df = pd.DataFrame({'label': roi_names})
    roi_df['laterality'] = [name.split('_')[-1] for name in roi_names]
    roi_df['category'] = [name.split('_')[0] for name in roi_names]
    roi_df = convert_roi_name_to_disp(roi_df)
    return roi_df



def convert_network_name(name):
    if name.startswith("Vis"): return "Vis"
    elif name.startswith("SomMot"): return "SomMot"
    elif name.startswith("DorsAttn"): return "DorsAttn"
    elif name.startswith("SalVen"): return "SalVentAttn"
    elif name.startswith("Limbic"): return "Limbic"
    elif name.startswith("Cont"): return "Cont"
    elif name.startswith("Default"): return "Default"
    else: return name


def get_roi_df(roi_path, fsn='yeo7'):
    roi_df = pd.read_csv(roi_path)
    row_id = roi_df[roi_df['Network']=='subcortex'].index.values
    roi_df.loc[row_id,'Network'] = roi_df[roi_df['Network']=='subcortex']['Anatomy'].values
    roi_df['Category'] = "cortex"
    roi_df.loc[row_id,'Category'] = "subcortex"
    if fsn=='yeo7':
        roi_df['Network'] = roi_df['Network'].apply(lambda x: convert_network_name(x))
    return roi_df[['ROI Name', 'Anatomy', 'Network', 'Laterality', 'Category']]
