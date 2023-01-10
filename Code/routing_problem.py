# %%
import gurobi as gp
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans

import openpyxl
import os
import sys

import plotly.graph_objects as go
import plotly.express as px


# %%####
# PLOTLY TEMPLATE
plotly_template = {"layout": go.Layout(font={"family": "Nunito",
                                             "size": 12,
                                             "color": "#707070", },
                                       yaxis={'zeroline': False, },
                                       xaxis={'zeroline': False, },
                                       plot_bgcolor="#ffffff",
                                       paper_bgcolor="#ffffff",
                                       colorway=px.colors.qualitative.Dark2,)}


# DATA
data_path = os.path.realpath(os.path.join('__file__', '..', 'Data'))
wb = openpyxl.load_workbook(f"{data_path}/data.xlsx", data_only=True)
w_cb = openpyxl.load_workbook(os.path.realpath(os.path.join('__file__', "..", "Data", "data.xlsx")))
sys.stdout = open(f'{data_path}/results_tsp/logging.txt', 'wt')


# WIND FARM DATA
wdat = pd.read_excel(f"{data_path}/data.xlsx", sheet_name='WindFarm')
wdat = wdat[['State Code', 'Size', 'Blades', 'Longitude', 'Latitude']]
wdat = wdat.rename(columns={'State Code': 'province', 'Size': 'size', 'Blades': 'blade_waste',
                   'Longitude': 'y', 'Latitude': 'x'})
wdat = wdat.reset_index().rename(columns={"index": "wind_farm"})
wdat['blade_waste'] = wdat['blade_waste'] / 20
wdat = wdat.query('province == "ON"')


# WIND FARM SIZES POINTS
scaler = preprocessing.MinMaxScaler(feature_range=(1, 10))
wdat['size'] = scaler.fit_transform(wdat[['size']])


# SETS
W = wdat['wind_farm'].to_list()


# MISC DATA
dco = 29.5
tco = 1000000
tca = 300
ph = 50
dk = 111


# %% #### MODELLING
print(f"MODELLING")
clusters_to_check = 0
to_break = False
kmeans = True


# MODELLING
for tc in range(5):

    clusters_to_check = tc
    to_break = True
    print(f"\tCluster Amount: {clusters_to_check +1}")
    C = [i for i in range(clusters_to_check + 1)]

    # SCALING
    scaler = preprocessing.StandardScaler()
    w_dat = wdat[['x', 'y', 'blade_waste', 'size']]
    w_loc_tf = pd.DataFrame(scaler.fit_transform(w_dat), columns=w_dat.columns)

    # CLUSTERING
    clusterer = KMeans(n_clusters=len(C))
    w_loc_label = clusterer.fit_predict(w_loc_tf[['x', 'y']])
    w_dat['cluster'] = w_loc_label

    # OPTMIZATION
    cluster_travel_cost = []
    cluster_fixed_cost = len(C) * tco
    cluster_travel = []
    cluster_req_cap = []
    cluster_req_met = []

    # RESULTS
    path_dat = {'from': [], 'to': [], 'move': []}

    for c in C:

        # SPLIT DATA
        w_cdat = w_dat.query(f"cluster == {c}")
        w_cx = w_cdat.to_dict()['x']
        w_cy = w_cdat.to_dict()['y']
        w_cb = w_cdat.to_dict()['blade_waste']
        Wc = list(w_cx.keys())

        # GENERATE DISTANCE
        wdi = {}
        for w1 in Wc:
            for w2 in Wc:
                wdi[w1, w2] = np.abs(w_cx[w1] - w_cx[w2]) + np.abs(w_cy[w1] - w_cy[w2])

        # OPTIMIZATION MODEL
        model = gp.Model('model')
        model.Params.TimeLimit = 60*5/(clusters_to_check+1)
        model.Params.LogToConsole = 1

        # DECISION VARIABLES
        vt = model.addVars(Wc, Wc, vtype=gp.GRB.BINARY, name='var_travel_from_to')
        vu = model.addVars(Wc, vtype=gp.GRB.INTEGER, name='var_MTZ_form')

        # BASIC CONSTRAINTS
        c_en = model.addConstrs((gp.quicksum(vt[w1, w2] for w1 in Wc) == 1 for w2 in Wc), name='con_enter_once')
        c_ex = model.addConstrs((gp.quicksum(vt[w1, w2] for w2 in Wc) == 1 for w1 in Wc), name='con_exit_once')
        c_es = model.addConstrs((vt[w1, w1] == 0 for w1 in Wc), name='con_enter_self')

        # MTZ CONSTRAINTS
        c_mtz_1 = model.addConstrs((vu[s1] - vu[s2] + len(Wc)*vt[s1, s2]
                                    <= len(Wc)-1
                                    for s1 in Wc[1:] for s2 in Wc[1:] if s1 != s2),
                                   name='con_MTZ_1')
        c_mtz_2 = model.addConstrs((vu[s1] >= 1 for s1 in Wc[1:]), name='con_MTZ_2')
        c_mtz_3 = model.addConstrs((vu[s1] <= len(Wc)-1 for s1 in Wc[1:]), name='con_MTZ_3')

        # OBJECTIVE
        tot_dist = gp.quicksum(wdi[(w1, w2)] * vt[(w1, w2)] for w1 in Wc for w2 in Wc)
        tot_cost = gp.quicksum(wdi[(w1, w2)]
                               * 50 * 29.5 * vt[(w1, w2)] * dk
                               / 1000000
                               for w1 in Wc for w2 in Wc)
        model.setObjective(tot_dist, sense=gp.GRB.MINIMIZE)
        model.optimize()

        # SUMMARY STATISTICS
        cluster_travel_cost.append(round(tot_cost.getValue()*1000000, 2))
        cluster_travel.append(round(tot_dist.getValue(), 2))
        cluster_req_cap.append(round(sum(w_cb.values()), 2))
        if 300 > sum(w_cb.values()):
            cluster_req_met.append(True)
        else:
            cluster_req_met.append(False)
            # clusters_to_check += 1
            # to_break = False

        # SAVE PATH
        for var in vt:
            path_dat['from'].append(var[0])
            path_dat['to'].append(var[1])
            path_dat['move'].append(vt[var].X)

    # LOGGING
    print('\n'*10)
    print(f"Clustering Travel Cost over 50 years: ${sum(cluster_travel_cost):,.2f}")
    print(f"Clustering Fixed Cost: ${cluster_fixed_cost:,.2f}")
    print(f"Clustering Total Cost: ${sum(cluster_travel_cost)+cluster_fixed_cost:,.2f}, Total Travel (km): {cluster_travel}")
    print(f"Cluster Requirements (blades): {cluster_req_cap}, Requirements Met: {cluster_req_met}")
    print('\n'*10)

    # FIX DATA
    path_df = pd.DataFrame(path_dat)
    path_df = path_df.join(w_dat[['x', 'y']], on='from').rename(columns={'x': 'from_x', 'y': 'from_y'})
    path_df = path_df.join(w_dat[['x', 'y']], on='to').rename(columns={'x': 'to_x', 'y': 'to_y'})
    w_dat['cluster'] = w_dat['cluster']+1
    w_dat['cluster'] = "Cluster: " + w_dat['cluster'].astype(str)
    w_dat = w_dat.rename(columns={'cluster': "Cluster"}).sort_values("Cluster")

    # SAVE DATA
    path_df.to_csv(f'{data_path}/results_tsp/CLUSTERS{clusters_to_check+1}_assignment.csv', index=False)
    w_dat.to_csv(f'{data_path}/results_tsp/CLUSTERS{clusters_to_check+1}_location.csv', index=False)

    # BOUNDS
    wx = wdat[['x']].to_dict()['x']
    wy = wdat[['y']].to_dict()['y']
    x_min = np.floor(np.min(np.array(list(wx.values()))))
    x_max = np.ceil(np.max(np.array(list(wx.values()))))
    y_min = np.floor(np.min(np.array(list(wy.values()))))
    y_max = np.ceil(np.max(np.array(list(wy.values()))))

    # SAVE PLOT
    fig_on = px.scatter_geo(w_dat, lat='y', lon='x',
                            color='Cluster',
                            size='size', size_max=10,
                            template=plotly_template,)

    # LINES PLOT
    for row in path_df.query('move == 1').itertuples():
        fig_on.add_trace(go.Scattergeo(
            mode="lines",
            lat=[row.from_y, row.to_y],
            lon=[row.from_x, row.to_x],
            showlegend=False,
            line={'color': '#969696', 'width': 0.5}))

    # MODIFYING FIGURE
    fig_on.update_geos(visible=False, resolution=50,
                       scope='north america',
                       showcountries=True, countrycolor="black",
                       showlakes=True, lakecolor='lightcyan',
                       showrivers=True, rivercolor='lightcyan',
                       showocean=True, oceancolor='lightcyan',
                       showland=True, landcolor='floralwhite',
                       showsubunits=True, subunitcolor="grey", subunitwidth=0.5,
                       lataxis={"range": [y_min, y_max]},
                       lonaxis={"range": [x_min, x_max]})
    fig_on.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                         legend={"yanchor": "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01,
                                 'borderwidth': 2, 'bordercolor': 'black'})

    fig_on.show(renderer='browser')
    fig_on.write_image(f'{data_path}/../Manuscripts/Report/graphics/fig_cluster{clusters_to_check+1}.jpeg', width=1000, height=800, scale=10)
