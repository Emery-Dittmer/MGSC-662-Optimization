# %%#### PACKAGES
import gurobi as gp
import pandas as pd
import numpy as np
from sklearn import preprocessing

import openpyxl
import os
import sys

import plotly.graph_objects as go
import plotly.express as px

# %%#### PLOTLY TEMPLATES
plotly_template = {"layout": go.Layout(font={"family": "Nunito",
                                             "size": 12,
                                             "color": "#707070", },
                                       yaxis={'zeroline': False, },
                                       xaxis={'zeroline': False, },
                                       plot_bgcolor="#ffffff",
                                       paper_bgcolor="#ffffff",
                                       colorway=px.colors.qualitative.Dark2,)}

# FULL DATA
data_path = os.path.realpath(os.path.join('__file__', '..', 'Data'))
wb = openpyxl.load_workbook(f"{data_path}/data.xlsx", data_only=True)
sys.stdout = open(f'{data_path}/results_fl/logging.txt', 'wt')

# SETS
ws_sets = wb['Sets']
T = [i+1 for i in range(ws_sets['B2'].value)]


# WIND FARM DATA
wdat = pd.read_excel(f"{data_path}/data.xlsx", sheet_name='WindFarm')
wdat = wdat[['State Code', 'Size', 'Deliveries_Yearly_Rounded', 'Waste_Delivery', 'Yearly_Total_Waste', 'Longitude', 'Latitude']]
wdat = wdat.rename(columns={'State Code': 'province', 'Size': 'size', 'Deliveries_Yearly_Rounded': 'deliv',
                   'Waste_Delivery': 'deliv_waste', 'Yearly_Total_Waste': 'total_waste', 'Longitude': 'y', 'Latitude': 'x'})
wdat = wdat.reset_index().rename(columns={"index": "wind_farm"})


# WIND FARM SIZES POINTS
scaler = preprocessing.MinMaxScaler(feature_range=(1, 10))
wdat['size'] = scaler.fit_transform(wdat[['size']])


# FACILITY DATA
ws_f = wb['FacilityData']
co = {t: ws_f[f"C{t+1}"].value for t in T}
ca = {t: ws_f[f"B{t+1}"].value for t in T}


# MISC DATA
dc = 29.5
ph = 50
dk = 111


wb.close()


def bound_generation(wind_farm_dat):

    # DATA
    wx = wind_farm_dat[['x']].to_dict()['x']
    wy = wind_farm_dat[['y']].to_dict()['y']
    wd = wind_farm_dat[['deliv']].to_dict()['deliv']
    ww = wind_farm_dat[['deliv_waste']].to_dict()['deliv_waste']

    # BOUNDS
    x_min = np.floor(np.min(np.array(list(wx.values()))))
    x_max = np.ceil(np.max(np.array(list(wx.values()))))
    x_diff = x_max - x_min

    y_min = np.floor(np.min(np.array(list(wy.values()))))
    y_max = np.ceil(np.max(np.array(list(wy.values()))))
    y_diff = y_max - y_min
    ds_diff = x_diff + y_diff

    d_min = np.floor(np.min(np.array(list(wd.values()))))
    d_max = np.ceil(np.max(np.array(list(wd.values()))))
    d_mean = np.mean(np.array(list(wd.values())))
    d_diff = d_max - d_min

    # RETURN
    return ((x_min, x_max, x_diff),
            (y_min, y_max, y_diff, ds_diff),
            (d_min, d_max, d_mean, d_diff))


def model_formulation_solving(wind_farm_dat, time_lim, bounds, logging=False):

    # CONVERT DATA
    W = wind_farm_dat['wind_farm'].to_list()

    wx = wind_farm_dat[['x']].to_dict()['x']
    wy = wind_farm_dat[['y']].to_dict()['y']
    wd = wind_farm_dat[['deliv']].to_dict()['deliv']
    ww = wind_farm_dat[['deliv_waste']].to_dict()['deliv_waste']

    x_min = bounds[0][0]
    x_max = bounds[0][1]
    x_dif = bounds[0][2]

    y_min = bounds[1][0]
    y_max = bounds[1][1]
    y_diff = bounds[1][2]
    ds_diff = bounds[1][3]

    d_min = bounds[2][0]
    d_max = bounds[2][1]
    d_mean = bounds[2][2]
    d_diff = bounds[2][3]

    # MODEL
    model = gp.Model('model')
    model.Params.LogToConsole = logging

    model.Params.Heuristics = 0.001
    model.Params.CutPasses = 3
    model.Params.PreDepRow = 0
    model.Params.PreQLinearize = 1

    model.Params.TimeLimit = time_lim

    # DECISION VARS
    fb = model.addVars(F, T, vtype=gp.GRB.BINARY, name='var_build_facility_FT')
    fx = model.addVars(F, T, ub=x_max, lb=x_min, name='var_facility_lat_FT')
    fy = model.addVars(F, T, ub=y_max, lb=y_min, name='var_facility_lon_FT')

    dist_x = model.addVars(W, F, T, ub=ds_diff, lb=-ds_diff, name='var_facility_distx_WFT')
    dist_y = model.addVars(W, F, T, ub=ds_diff, lb=-ds_diff, name='var_facility_disty_WFT')
    dist = model.addVars(W, F, T, ub=ds_diff, lb=-ds_diff, name='var_facility_dist_WFT')
    wa = model.addVars(W, F, T, vtype=gp.GRB.BINARY, name='var_wind_farm_assignment')
    dw = model.addVars(W, F, T, vtype=gp.GRB.INTEGER, ub=d_max, name='var_deliver_waste_WFT')

    # DEMAND
    c_d = model.addConstrs((gp.quicksum(wa[(w, f, t)] for t in T for f in F) == 1 for w in W), name='con_demand')

    # CAPACITY
    c_c = model.addConstrs((gp.quicksum(wa[(w, f, t)] * ww[w] * wd[w] for w in W)
                            <= ca[t] * fb[(f, t)] for f in F for t in T),
                           name='con_capacity')

    # DISTANCE (Manhattan)
    c_dx_1 = model.addConstrs((dist_x[(w, f, t)] >= fx[(f, t)] - wx[w] for w in W for f in F for t in T), name='con_dist_x1')
    c_dx_2 = model.addConstrs((dist_x[(w, f, t)] >= -(fx[(f, t)] - wx[w]) for w in W for f in F for t in T), name='con_dist_x2')
    c_dy_1 = model.addConstrs((dist_y[(w, f, t)] >= fy[(f, t)] - wy[w] for w in W for f in F for t in T), name='con_dist_y1')
    c_dx_2 = model.addConstrs((dist_y[(w, f, t)] >= -(fy[(f, t)] - wy[w]) for w in W for f in F for t in T), name='con_dist_y2')
    c_dist = model.addConstrs((dist[(w, f, t)] == dist_x[(w, f, t)] + dist_y[(w, f, t)] for w in W for f in F for t in T), name='con_dist')

    # OBJECTIVES
    del_cost_relax = gp.quicksum(dist[(w, f, t)] * wa[(w, f, t)] * wd[w] * dc * dk * 50 / 1000000 for w in W for f in F for t in T)
    del_cost_full = gp.quicksum(dist[(w, f, t)] * dw[(w, f, t)] * dc * dk * 50 / 1000000 for w in W for f in F for t in T)
    build_cost = gp.quicksum(co[t] * fb[(f, t)] / 1000000 for f in F for t in T)

    # RELAXED OPTIMIZATION
    model.setObjective(del_cost_relax + build_cost, sense=gp.GRB.MINIMIZE)
    model.optimize()

    for var in fb:
        fb[var].ub = fb[var].X
        fb[var].lb = fb[var].X

    for var in fx:
        fx[var].ub = fx[var].X
        fx[var].lb = fx[var].X

    for var in fy:
        fy[var].ub = fy[var].X
        fy[var].lb = fy[var].X

    for var in dist:
        dist[var].ub = dist[var].X
        dist[var].lb = dist[var].X

    for var in wa:
        wa[var].ub = wa[var].X
        wa[var].lb = wa[var].X

    # DETAILED OPTIMIZATION
    model.Params.LogToConsole = 0
    model.Params.NonConvex = 2
    model.Params.MIPGap = 0.0001

    model.remove(c_d)
    model.remove(c_c)
    c_d = model.addConstrs((gp.quicksum(dw[(w, f, t)] for t in T for f in F) == wd[w] for w in W), name='con_demand')
    c_c = model.addConstrs((gp.quicksum(dw[(w, f, t)] * ww[w] for w in W)
                            <= ca[t] * fb[(f, t)] for f in F for t in T),
                           name='con_capacity')

    model.setObjective(del_cost_full + build_cost, sense=gp.GRB.MINIMIZE)
    model.optimize()

    # results
    facil_placement = {'fac_id': [], 'fac_count': [], 'fac_type': [],
                       'fac_build': [], 'fac_x': [], 'fac_y': []}
    for var in fb:
        facil_placement['fac_id'].append(var)
        facil_placement['fac_count'].append(var[0])
        facil_placement['fac_type'].append(var[1])
        facil_placement['fac_build'].append(round(fb[var].X, 0))
        facil_placement['fac_x'].append(fx[var].X)
        facil_placement['fac_y'].append(fy[var].X)

    wind_deliveries = {'wind_farm': [], 'fac_id': [],
                       'deliv': []}
    for var in wa:
        wind_deliveries['wind_farm'].append(var[0])
        wind_deliveries['fac_id'].append((var[1], var[2]))
        wind_deliveries['deliv'].append(round(wa[var].X, 0))

    return (facil_placement,
            wind_deliveries,
            (del_cost_full.getValue(), build_cost.getValue()))


def plot_generation(wind_farm_data, fac_place_data, deliv_data, bounds):

    # BOUNDS
    x_min = bounds[0][0]
    x_max = bounds[0][1]
    y_min = bounds[1][0]
    y_max = bounds[1][1]

    # WIND FARM LOCATIONS
    loc_wf = wind_farm_data[['wind_farm', 'size', 'x', 'y']]
    loc_wf['Facility'] = 'Wind Farm'
    loc_wf = loc_wf.rename(columns={'wind_farm': 'id'})

    # RELEVANT FACILITY LOCATIONS
    loc_fc = fac_place_data.query('fac_build == 1')
    loc_fc = loc_fc[['fac_id', 'fac_x', 'fac_y', 'fac_type']].rename(columns={'fac_id': 'id', 'fac_x': 'x', 'fac_y': 'y', 'fac_type': 'Facility'})
    loc_fc['Facility'] = "Facility Type " + loc_fc['Facility'].astype(str)
    loc_fc['size'] = 15

    # ALL LOCATIONS
    loc = pd.concat([loc_wf, loc_fc], ignore_index=True)
    loc = loc.sort_values('Facility', ascending=False)

    # CONNECTIONS
    con = deliv_data.query('deliv == 1').join(wind_farm_data[['wind_farm', 'x', 'y']].set_index('wind_farm'), on='wind_farm', how='left')
    con = con.rename(columns={'x': 'from_x', 'y': 'from_y'})
    con = con.join(fac_place_data.query('fac_build == 1')[['fac_id', 'fac_x', 'fac_y']].set_index('fac_id'), on='fac_id', how='left')
    con = con.rename(columns={'fac_x': 'to_x', 'fac_y': 'to_y'})

    # LOCATIONS PLOT
    fig_on = px.scatter_geo(loc, lat='y', lon='x',
                            color='Facility',
                            size='size', size_max=10,
                            template=plotly_template,)

    # LINES PLOT
    for row in con.itertuples():
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

    # REORDERING TRACES
    fig_traces = fig_on.data
    fig_traces_new = []
    for i in range(len(T)+1, len(fig_traces)):
        fig_traces_new.append(fig_traces[i])

    for i in range(len(T)+1):
        fig_traces_new.append(fig_traces[i])

    fig_on.data = fig_traces_new
    return (fig_on)


# %% IMPORT RESULTS - ON
wdat_ON = wdat.query('province == "ON"')
fp_df_ON = pd.read_csv(f'{data_path}/results_fl_safe/ON_location.csv')
wd_df_ON = pd.read_csv(f'{data_path}/results_fl_safe/ON_assignment.csv')
bounds_ON = bound_generation(wdat_ON)
fig_on = plot_generation(wdat_ON, fp_df_ON, wd_df_ON, bounds_ON)
fig_on.write_image(f'{data_path}/../Manuscripts/Report/graphics/fig_on.jpeg', width=1000, height=800, scale=10)
fig_on

# %% IMPORT RESULTS - CAN
fp_df_CAN = pd.read_csv(f'{data_path}/results_fl_safe/CAN_location.csv')
wd_df_CAN = pd.read_csv(f'{data_path}/results_fl_safe/CAN_assignment.csv')
bounds_CAN = bound_generation(wdat)
fig_can = plot_generation(wdat, fp_df_CAN, wd_df_CAN, bounds_CAN)
fig_can.write_image(f'{data_path}/../Manuscripts/Report/graphics/fig_can.jpeg', width=1000, height=450, scale=10)
fig_can

# %% IMPORT RESULTS - PBP
fp_df_PBP = pd.read_csv(f'{data_path}/results_fl_safe/PBP_location.csv')
wd_df_PBP = pd.read_csv(f'{data_path}/results_fl_safe/PBP_assignment.csv')
bounds_PBP = bound_generation(wdat)
fig_pbp = plot_generation(wdat, fp_df_PBP, wd_df_PBP, bounds_PBP)
fig_pbp.write_image(f'{data_path}/../Manuscripts/Report/graphics/fig_pbp.jpeg', width=1000, height=450, scale=10)
fig_pbp

# %%####
# ONTARIO ONLY
print("ONTARIO ONLY")
wdat_ON = wdat.query('province == "ON"')
F = [i+1 for i in range(4)]


print("\n"*5)
bounds_ON = bound_generation(wdat_ON)
facil_placement, wind_deliveries, costs = model_formulation_solving(wdat_ON, 60*60*1, bounds_ON, True)
print("\n"*5)
fp_df = pd.DataFrame(facil_placement)
wd_df = pd.DataFrame(wind_deliveries)


print('\tOntario Only')
print(f"Delivery Cost (MIL): ${costs[0]:,.2f}")
print(f"Build Cost (MIL): ${costs[1]:,.2f}")
print(f"Total Cost (MIL): ${(costs[0]+costs[1]):,.2f}")
print("\n"*5)


fig_on = plot_generation(wdat_ON, fp_df, wd_df, bounds_ON)
fig_on.write_image(f'{data_path}/../Manuscripts/Report/graphics/fig_on.jpeg', width=1000, height=800, scale=10)
fp_df.to_csv(f'{data_path}/results_fl/ON_location.csv', index=False)
wd_df.to_csv(f'{data_path}/results_fl/ON_assignment.csv', index=False)


# %%####
# CANADA WIDE
print("CANADA WIDE")
F = [i+1 for i in range(6)]


print("\n"*5)
bounds_CAN = bound_generation(wdat)
facil_placement, wind_deliveries, costs = model_formulation_solving(wdat, 60*60*1, bounds_CAN, True)
print("\n"*5)
fp_df = pd.DataFrame(facil_placement)
wd_df = pd.DataFrame(wind_deliveries)


print('\tCanada Wide')
print(f"Delivery Cost (MIL): ${costs[0]:,.2f}")
print(f"Build Cost (MIL): ${costs[1]:,.2f}")
print(f"Total Cost (MIL): ${(costs[0]+costs[1]):,.2f}")
print("\n"*5)


fig_can = plot_generation(wdat, fp_df, wd_df, bounds_CAN)
fig_can.write_image(f'{data_path}/../Manuscripts/Report/graphics/fig_can.jpeg', width=1000, height=450, scale=10)
fp_df.to_csv(f'{data_path}/results_fl/CAN_location.csv', index=False)
wd_df.to_csv(f'{data_path}/results_fl/CAN_assignment.csv', index=False)


# %%####
# PROVINCE BY PROVINCE
print("PROVINCE BY PROVINCE")
F = [i+1 for i in range(4)]
fp_df = pd.DataFrame()
wd_df = pd.DataFrame()
costs_t = [0, 0]


for k, wdat_p in wdat.groupby('province'):
    print("\n"*5)
    bounds_PV = bound_generation(wdat_p)
    facil_placement, wind_deliveries, costs = model_formulation_solving(wdat_p, 60*60/10, bounds_PV, True)
    print("\n"*5)
    fp_df_i = pd.DataFrame(facil_placement)
    fp_df_i['prov'] = k
    wd_df_i = pd.DataFrame(wind_deliveries)
    wd_df_i['prov'] = k

    fp_df = pd.concat([fp_df, fp_df_i], ignore_index=True)
    wd_df = pd.concat([wd_df, wd_df_i], ignore_index=True)

    print(f"\tProvince {k}")
    print(f"Delivery Cost (MIL): ${costs[0]:,.2f}")
    print(f"Build Cost (MIL): ${costs[1]:,.2f}")
    print(f"Total Cost (MIL): ${(costs[0]+costs[1]):,.2f}")
    print("\n"*5)

    costs_t[0] += costs[0]
    costs_t[1] += costs[1]


fp_df['fac_id'] = fp_df['fac_id'].astype(str) + " " + fp_df['prov']
wd_df['fac_id'] = wd_df['fac_id'].astype(str) + " " + wd_df['prov']


print('OVERALL')
print(f"Delivery Cost (MIL): ${costs_t[0]:,.2f}")
print(f"Build Cost (MIL): ${costs_t[1]:,.2f}")
print(f"Total Cost (MIL): ${(costs_t[0]+costs_t[1]):,.2f}")
print("\n"*5)


fig_pbp = plot_generation(wdat, fp_df, wd_df, bounds_CAN)
fig_pbp.write_image(f'{data_path}/../Manuscripts/Report/graphics/fig_pbp.jpeg', width=1000, height=450, scale=10)
fp_df.to_csv(f'{data_path}/results_fl/PBP_location.csv', index=False)
wd_df.to_csv(f'{data_path}/results_fl/PBP_assignment.csv', index=False)


# %%
