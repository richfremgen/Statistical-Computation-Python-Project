import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from functions.passing_data_fxns import get_summary_pass_data


def draw_soccer_pitch(figsize=(9, 6)):
    """
    Function that plots a scaled soccer pitch of length 120*90 metres 
    
    https://www.kaggle.com/code/ajsteele/draw-soccer-pitch-with-matplotlib/notebook
    """
    rect = patches.Rectangle((-1, -1), 122, 92, linewidth=0.1,
                             edgecolor='r', facecolor='black', zorder=0, label='_nolegend_')

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)
    # Main pitch markings, ie sidelines, penalty area and halfway line
    plt.plot([0, 0,  0, 120, 120, 0,     0,  16.5,  16.5,     0,     0,   5.5,   5.5, 
                  0,  0, 60, 60, 120,   120, 103.5, 103.5,   120,   120, 114.5, 114.5,   120], 
             [0, 0, 90,  90,   0, 0, 25.85, 25.85, 66.15, 66.15, 55.15, 55.15, 36.85, 
              36.85, 90, 90,  0,   0, 25.85, 25.85, 66.15, 66.15, 55.15, 55.15, 36.85, 36.85], color='white')
    
    # Secondary pitch markings, ie penalty spots, centre circle etc
    plt.plot([11, 11.5],[45, 45], color='white')
    plt.plot([109, 108.5],[45, 45], color='white')
    
    centre_circle = patches.Circle([60, 45], 9.15, edgecolor='white', facecolor='black')
    ax.add_patch(centre_circle)
    
    left_arc = patches.Arc([16.5, 45], 9.15, 16, theta1=270.0, theta2=90.0, color='white')
    ax.add_patch(left_arc)
    right_arc = patches.Arc([103.5, 45], 9.15, 16, theta1=90.0, theta2=270.0, color='white')
    ax.add_patch(right_arc)
    
    bl_corner = patches.Arc([0, 0], 2.5, 2.5, theta1=0.0, theta2=90.0, color='white')
    tl_corner = patches.Arc([0, 90], 2.5, 2.5, theta1=270.0, color='white')
    br_corner = patches.Arc([120, 0], 2.5, 2.5, theta1=90.0, theta2=180.0, color='white')
    tr_corner = patches.Arc([120, 90], 2.5, 2.5, theta1=180.0, theta2=270.0,color='white')
    ax.add_patch(bl_corner)
    ax.add_patch(tl_corner)
    ax.add_patch(br_corner)
    ax.add_patch(tr_corner)
    
    plt.xlim(-1, 121)
    plt.ylim(-1, 91)
    plt.axis('off')    

    return fig, ax


# heatmap of pass counts by origin position 
def plot_pass_length_heatmap(p):
    # make overall average by bins 
    or_pbd = p[['origin_x', 'origin_y', 'distance', 'country']]
    or_pbdc = or_pbd.drop(['country'], axis=1)
    or_cuts = pd.DataFrame({str(feature) + 'Bin' : pd.cut(or_pbdc[feature], 40) for feature in ['origin_x', 'origin_y']})
    or_ov_avg = or_pbdc.join(or_cuts).groupby( list(or_cuts) ).mean()
    or_ov_avg = or_ov_avg.unstack(level = 0) 
    or_ov_avg = or_ov_avg.iloc[::-1]

    # make average by country dict by bins 
    origin_mean_dist_data = {}
    for country in or_pbd.country.unique():
        pbdc = or_pbd[or_pbd.country == country].drop(['country'], axis=1)
        cuts = pd.DataFrame({str(feature) + 'Bin' : pd.cut(pbdc[feature], 40) for feature in ['origin_x', 'origin_y']})
        means = pbdc.join(cuts).groupby( list(cuts) ).mean()
        means = means.unstack(level = 0) 
        means = means.iloc[::-1]
        means_test = means - or_ov_avg
        origin_mean_dist_data.update({country:means_test})
        
    # graph
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    cbar_ax = fig.add_axes([.91, .15, .03, .7])
    f1 = sns.heatmap(
        origin_mean_dist_data['Spain'][['distance']], vmin=-10, vmax=15, ax=axes[0, 0], cbar=False, center=0
    )
    f1.set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None, title='Spain')
    f2 = sns.heatmap(
        origin_mean_dist_data['Italy'][['distance']], vmin=-10, vmax=15, ax=axes[0, 1], cbar=False, center=0
    )
    f2.set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None, title='Italy')
    f3 = sns.heatmap(
        origin_mean_dist_data['France'][['distance']], vmin=-10, vmax=15, ax=axes[0, 2], cbar=False, center=0
    )
    f3.set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None, title='France')
    f4 = sns.heatmap(
        origin_mean_dist_data['England'][['distance']], vmin=-10, vmax=15, ax=axes[1, 0], cbar=False, center=0
    )
    f4.set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None, title='England')
    f5 = sns.heatmap(
        origin_mean_dist_data['Germany'][['distance']], vmin=-10,
        vmax=15, ax=axes[1, 1], center=0, cbar_ax=cbar_ax
    )
    f5.set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None, title='Germany')
    fig.delaxes(axes[1][2])
    
    fig.suptitle('Germany & England pass farther than average in the backfield, while France & Italy pass relatively shorter distances')
    fig.subplots_adjust(top=0.9)
    
    return None


# passes by country, colored by frequency 
def plot_pass_by_country(pp_avg):
    pp_avg_agg_c = pd.DataFrame(pp_avg.groupby(['country', 'or_avg_x', 'or_avg_y', 'dest_avg_x', 'dest_avg_y']).size()
            ).reset_index().rename({0:'count'}, axis=1)
    
    # make country prop
    pp_avg_agg_c['prop'] = pp_avg_agg_c['count'] / pp_avg_agg_c.groupby('country')['count'].transform('sum')
    pp_avg_agg_c['n_prop'] = pp_avg_agg_c['prop'] * 200000

    # draw 
    for country in pp_avg_agg_c.country.unique():
        print('Plotting', country, 'points')
        df = pp_avg_agg_c[(pp_avg_agg_c['country'] == country) & (pp_avg_agg_c['n_prop'] > 3)].copy()

        fig, ax = draw_soccer_pitch()

        for i in range(len(df)):
            if (df['n_prop'].iloc[i] < 5):
                lnw = 0.5
                col = 'white'
            elif (df['n_prop'].iloc[i] < 10):
                lnw = 1
                col = 'blue'
            else:
                lnw = 2
                col = 'red'

            plt.plot((df['or_avg_x'].iloc[i],df['dest_avg_x'].iloc[i]),
                     (df['or_avg_y'].iloc[i],df['dest_avg_y'].iloc[i]),
                     color=col, linewidth = lnw)
        
        plt.title(country)
        plt.legend(labels=['<50', '50-150', '>150'], loc=2)
        leg = ax.get_legend()
        hl_dict = {handle.get_label(): handle for handle in leg.legendHandles}
        hl_dict['_child1'].set_color('white')
        hl_dict['_child2'].set_color('blue')
        hl_dict['_child3'].set_color('red')
    
    return None


# passes by country, colored by position
def plot_pass_by_country_role(pp_avg):
    # get pass counts by groups
    pp_avg_agg = pp_avg.groupby(['country', 'position', 'or_avg_x', 'or_avg_y', 'dest_avg_x', 'dest_avg_y']).size()
    pp_avg_agg = pd.DataFrame(pp_avg_agg).reset_index().rename({0:'count'}, axis=1)

    # make country prop
    pp_avg_agg['prop'] = pp_avg_agg['count'] / pp_avg_agg.groupby(['country','position'])['count'].transform('sum')
    pp_avg_agg['n_prop'] = pp_avg_agg['prop'] * 50000

    # take out goalkeeper 
    pp_avg_agg_ng = pp_avg_agg[(pp_avg_agg['position'] != 'Goalkeeper')].copy()

    for country in pp_avg_agg_ng.country.unique():
        print('Plotting', country, 'points')
        df = pp_avg_agg_ng[(pp_avg_agg_ng['country'] == country) & (pp_avg_agg_ng['n_prop'] > 2)].copy()

        fig, ax = draw_soccer_pitch()

        for i in range(len(df)):
            if (df['n_prop'].iloc[i] < 5):
                lnw = 0.5
            elif (df['n_prop'].iloc[i] < 10):
                lnw = 1
            else:
                lnw = 2

            if(df['position'].iloc[i] == 'Defender'):
                col = 'white'
            elif(df['position'].iloc[i] == 'Midfielder'):
                col = 'red'
            else: 
                col = 'blue'

            plt.plot((df['or_avg_x'].iloc[i],df['dest_avg_x'].iloc[i]),
                     (df['or_avg_y'].iloc[i],df['dest_avg_y'].iloc[i]),
                     color=col, linewidth = lnw, alpha=0.75)
            
        plt.title(country)
        plt.legend(labels=['<50', '50-150', '>150'], loc=2)
        leg = ax.get_legend()
        hl_dict = {handle.get_label(): handle for handle in leg.legendHandles}
        hl_dict['_child1'].set_color('white')
        hl_dict['_child2'].set_color('blue')
        hl_dict['_child3'].set_color('red')
    
    return None


# graph directional movement based on most frequent pass
def plot_freq_pass(ps_df):
    # get 10x10 grid data 
    pp_avg_10 = get_summary_pass_data(ps_df, 10)
    # by country 
    pp_avg_agg_c_10 = pd.DataFrame(pp_avg_10.groupby(['country', 'or_avg_x', 'or_avg_y', 'dest_avg_x', 'dest_avg_y']).size()
                ).reset_index().rename({0:'count'}, axis=1)
    # make prop column
    pp_avg_agg_c_10['prop'] = pp_avg_agg_c_10['count'] / pp_avg_agg_c_10.groupby('country')['count'].transform('sum')
    pp_avg_agg_c_10['n_prop'] = pp_avg_agg_c_10['prop'] * 7000

    # find most freq pass by bin 
    idx = pp_avg_agg_c_10.groupby(['country', 'or_avg_x', 'or_avg_y'])['n_prop'].transform('max') == pp_avg_agg_c_10['n_prop']
    df_max_10 = pp_avg_agg_c_10[idx].copy()

    # graph
    for country in df_max_10.country.unique():
        print('Plotting', country, 'points')
        df = df_max_10[(df_max_10['country'] == country)].copy()

        fig, ax = draw_soccer_pitch()

        for i in range(len(df)):

            if (df['n_prop'].iloc[i] < 2):
                lnw = 0.5
                col = 'white'
            elif (df['n_prop'].iloc[i] < 5):
                lnw = 1
                col = 'blue'
            else:
                lnw = 2
                col = 'red'

            dx = (df['dest_avg_x'].iloc[i] - df['or_avg_x'].iloc[i])
            dy = (df['dest_avg_y'].iloc[i] - df['or_avg_y'].iloc[i])

            plt.arrow(
                x=df['or_avg_x'].iloc[i], y=df['or_avg_y'].iloc[i], dx=dx,
                dy=dy, width=.5, color=col, alpha=0.75
            ) 
            
        plt.title(country)    
        plt.legend(labels=['<50', '50-150', '>150'], loc=2)
        leg = ax.get_legend()
        hl_dict = {handle.get_label(): handle for handle in leg.legendHandles}
        hl_dict['_child1'].set_color('white')
        hl_dict['_child2'].set_color('blue')
        hl_dict['_child3'].set_color('red')
    
    
    return None

