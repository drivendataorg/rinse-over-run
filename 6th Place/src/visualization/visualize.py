#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set(style="white")


def plot_process(g):
    assert g['process_id'].nunique() == 1
    g = g.copy()
    process_id, object_id, pipeline = g.iloc[0].process_id, g.iloc[0].object_id, g.iloc[0].pipeline
    
    fig, axs = plt.subplots(figsize=(15, 20), nrows=8, sharex=True)
    g[['total_turbidity_liter']].plot(ax=axs[0])
    y = axs[0].get_ylim()[1]

    for p in range(1, 7):
        g.loc[g[g['phase_num']==p].index, f'phase_num_{p}'] = y
        g[[f'phase_num_{p}']].plot(kind='line', ax=axs[0], linewidth=5)
    axs[0].set_ylabel("total_turbidity_liter")
    
    g[['tank_temperature_pre_rinse',
       'tank_temperature_caustic', 'tank_temperature_acid', 'return_temperature']].plot(ax=axs[1])
    
    g[['tank_concentration_caustic', 'tank_concentration_acid']].plot(ax=axs[2])
    
    g[['return_conductivity']].plot(ax=axs[3])
    
    bool_columns = ['supply_pump', 'supply_pre_rinse', 'supply_caustic', 'return_caustic', 
                    'supply_acid', 'return_acid', 'supply_clean_water', 'return_recovery_water', 
                    'return_drain',
                    'tank_lsh_caustic', 'tank_lsh_clean_water', 'tank_lsh_pre_rinse',
                    'object_low_level',
                    ]
    for i, c in enumerate(bool_columns):
        s = g[c].map(lambda x: i if x is True else None)
        if all(s.isnull()):
            s.iloc[0] = i
        s.plot(kind='line', ax=axs[4], linewidth=2)
    axs[4].set_yticks(range(len(bool_columns)))
    axs[4].set_yticklabels(bool_columns)
    
    g[['supply_pressure']].plot(ax=axs[5])
    g[['supply_flow', 'return_flow']].plot(ax=axs[6])

    g[['object_low_level', 'tank_level_pre_rinse', 'tank_level_caustic',
       'tank_level_acid', 'tank_level_clean_water']].plot(ax=axs[7])
    
    for ax in axs:
        sns.despine(ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.xaxis.grid(True, alpha=0.4)
    fig.suptitle(f"process_id={process_id} object_id={object_id} pipepline={pipeline}")

    return fig, axs
