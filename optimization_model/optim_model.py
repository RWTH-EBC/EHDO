# -*- coding: utf-8 -*-

"""

EHDO - ENERGY HUB DESIGN OPTIMIZATION Tool

Developed by:   E.ON Energy Research Center, 
                Institute for Energy Efficient Buildings and Indoor Climate, 
                RWTH Aachen University, 
                Germany
               
Contact:        Marco Wirtz 
                marco.wirtz@eonerc.rwth-aachen.de

"""

import gurobipy as gp
import numpy as np
import time

from django.conf import settings

def run_optim(devs, param, dem, result_dict, flags):


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load model parameters
    start_time = time.time()
    
    days = range(param["n_clusters"])
    time_steps = range(24)
    year = range(365)
    
    # Get sigma function which assigns every day of the year to a design day
    sigma = param["sigma"]

    # Create set for devices
    all_devs = ["PV", "WT", "STC", "WAT",
                "HP", "EB", "CC", "AC", 
                "CHP", "BOI", "GHP",
                "BCHP", "BBOI", "WCHP", "WBOI",
                "ELYZ", "FC", "H2S", "SAB",                
                "TES", "CTES", "BAT", "GS",  
                ]

                
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Set up model and create variables

    # Create a new model
    model = gp.Model("Energy_hub_model")
    
    # Purchase decision binary variables (1 if device is installed, 0 otherwise)
    x = {}
    for device in all_devs:
        x[device] = model.addVar(vtype="B", name="x_" + str(device))
        
    # Device's capacity (i.e. rated power)
    cap = {}
    for device in all_devs:
        cap[device] = model.addVar(vtype="C", name="nominal_capacity_" + str(device))
        
    # Roof area used for PV and solar thermal collector installation
    area = {}
    for device in ["PV", "STC"]:
        area[device] = model.addVar(vtype = "C", name="roof_area_" + str(device))
        
    # Gas flow to/from devices
    gas = {}
    for device in ["CHP", "BOI", "GHP", "SAB", "from_grid", "to_grid"]:
        gas[device] = {}
        for d in days:
            gas[device][d] = {}
            for t in time_steps:
                gas[device][d][t] = model.addVar(vtype="C", name="gas_" + device + "_d" + str(d) + "_t" + str(t))
    
    # Electric power to/from devices
    power = {}
    for device in ["PV", "WT", "WAT", "HP", "EB", "CC", "CHP", "BCHP", "WCHP", "ELYZ", "FC", "from_grid", "to_grid"]:
        power[device] = {}
        for d in days:
            power[device][d] = {}
            for t in time_steps:
                power[device][d][t] = model.addVar(vtype="C", name="power_" + device + "_d" + str(d) + "_t" + str(t))
       
    # Heat to/from devices
    heat = {}
    for device in ["STC", "HP", "EB", "AC", "CHP", "BOI", "GHP", "BCHP", "BBOI", "WCHP", "WBOI", "FC"]:
        heat[device] = {}
        for d in days:
            heat[device][d] = {}
            for t in time_steps:
                heat[device][d][t] = model.addVar(vtype="C", name="heat_" + device + "_d" + str(d) + "_t" + str(t))
    
    # Cooling power to/from devices
    cool = {}
    for device in ["CC", "AC"]:
        cool[device] = {}
        for d in days:
            cool[device][d] = {}
            for t in time_steps:
                cool[device][d][t] = model.addVar(vtype="C", name="cool_" + device + "_d" + str(d) + "_t" + str(t))
                
    # Hydrogen to/from devices
    hydrogen = {}
    for device in ["ELYZ", "FC", "SAB", "import"]:
        hydrogen[device] = {}
        for d in days:
            hydrogen[device][d] = {}
            for t in time_steps:
                hydrogen[device][d][t] = model.addVar(vtype="C", name="hydrogen_" + device + "_d" + str(d) + "_t" + str(t))
                
    # Biomass to devices
    biom = {}
    for device in ["BCHP", "BBOI", "import"]:
        biom[device] = {}
        for d in days:
            biom[device][d] = {}
            for t in time_steps:
                biom[device][d][t] = model.addVar(vtype="C", name="biom_" + device + "_d" + str(d) + "_t" + str(t))
                
    # Waste to devices
    waste = {}
    for device in ["WCHP", "WBOI", "import"]:
        waste[device] = {}
        for d in days:
            waste[device][d] = {}
            for t in time_steps:
                waste[device][d][t] = model.addVar(vtype="C", name="waste_" + device + "_d" + str(d) + "_t" + str(t))

    # Storage variables
    ch = {}  # Energy flow to charge storage device
    soc = {} # State of charge
    for device in ["TES", "CTES", "BAT", "H2S", "GS"]:
        ch[device] = {}
        soc[device] = {}
        for d in days:
            ch[device][d] = {}
            for t in time_steps:
                # For charge variable: ch is positive if storage is charged, and negative if storage is discharged
                ch[device][d][t] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="ch_" + device + "_d" + str(d) + "_t" + str(t))
        for day in year:
            soc[device][day] = {}
            for t in time_steps:
                soc[device][day][t] = model.addVar(vtype="C", name="soc_" + device + "_d" + str(day) + "_t" + str(t))

    # Variables for annual device costs     
    inv = {}
    c_inv = {}
    c_om = {}
    c_total = {}
    for device in all_devs:
        inv[device] = model.addVar(vtype = "C", name="investment_costs_" + device)
    for device in all_devs:
        c_inv[device] = model.addVar(vtype = "C", name="annual_investment_costs_" + device)
    for device in all_devs:
        c_om[device] = model.addVar(vtype = "C", name="om_costs_" + device)
    for device in all_devs:
        c_total[device] = model.addVar(vtype = "C", name="total_annual_costs_" + device)      

    # Capacity of grid connections (gas and electricity)
    grid_limit_el = model.addVar(vtype = "C", name="grid_limit_el")  
    grid_limit_gas = model.addVar(vtype = "C", name="grid_limit_gas")    
    
    # Total energy amounts taken from grid and fed into grid
    from_el_grid_total = model.addVar(vtype = "C", name="from_el_grid_total")
    to_el_grid_total = model.addVar(vtype = "C", name="to_el_grid_total")
    
    from_gas_grid_total = model.addVar(vtype = "C", name="from_gas_grid_total")
    to_gas_grid_total = model.addVar(vtype = "C", name="to_gas_grid_total")
    
    biom_import_total = model.addVar(vtype = "C", name="biom_import_total")
    waste_import_total = model.addVar(vtype = "C", name="waste_import_total")
    hydrogen_import_total = model.addVar(vtype = "C", name="hydrogen_import_total")
    
    # Total revenue for feed-in
    rev_feed_in_gas = model.addVar(vtype="C", name="rev_feed_in_gas")
    rev_feed_in_el = model.addVar(vtype="C", name="rev_feed_in_el")

    # Electricity/gas/biomass costs
    supply_costs_el = model.addVar(vtype = "C", name="supply_costs_el")    
    cap_costs_el = model.addVar(vtype = "C", name="cap_costs_el")    
    supply_costs_gas = model.addVar(vtype = "C", name="supply_costs_gas")    
    cap_costs_gas = model.addVar(vtype = "C", name="cap_costs_gas")    
    supply_costs_biom = model.addVar(vtype = "C", name="supply_costs_biomass")   
    supply_costs_waste = model.addVar(vtype = "C", name="supply_costs_waste")   
    supply_costs_hydrogen = model.addVar(vtype = "C", name="supply_costs_hydrogen")       
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Objective functions
    obj = {}
    obj["tac"] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="total_annualized_costs") 
    obj["co2"] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="total_CO2") 

    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Assign objective function
    model.update()
    model.setObjective((1-param["optim_focus"]) * obj["tac"]
                        + param["optim_focus"]  * obj["co2"], gp.GRB.MINIMIZE)
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Add constraints

    #%% Constraints defined by user in GUI
    
    for device in all_devs:        
        if devs[device]["feasible"] == True:
            model.addConstr(x[device] == 1)
        else:
            model.addConstr(x[device] == 0)
            
      
    #%% CONTINUOUS SIZING OF DEVICES: minimum capacity <= capacity <= maximum capacity
    
    for device in ["WT", "WAT", "HP", "EB", "CC", "AC", "CHP", "BOI", "GHP", "BCHP", "BBOI", "WCHP", "WBOI", "ELYZ", "FC", "H2S", "SAB", "TES", "CTES", "BAT", "GS"]:  # PV/STC is not listed due to min_area/max_area.
        model.addConstr(cap[device] >= x[device] * devs[device]["min_cap"])
        model.addConstr(cap[device] <= x[device] * devs[device]["max_cap"])
        
    for d in days:
        for t in time_steps:
            for device in ["STC", "EB", "HP", "BOI", "GHP", "BBOI", "WBOI"]:
                model.addConstr(heat[device][d][t] <= cap[device])

            for device in ["PV", "WT", "WAT", "CHP", "BCHP", "WCHP", "ELYZ", "FC"]:
                model.addConstr(power[device][d][t] <= cap[device])

            for device in ["CC", "AC"]:
                model.addConstr(cool[device][d][t] <= cap[device])
                
            for device in ["SAB"]:
                model.addConstr(gas[device][d][t] <= cap[device])
                
            # Limitation of power from and to grid   
            for device in ["from_grid", "to_grid"]:
                model.addConstr(power[device][d][t] <= grid_limit_el)
                model.addConstr(gas[device][d][t]   <= grid_limit_gas)    

            # PV and STC: minimum area < used roof area <= maximum area
            for device in ["PV", "STC"]:            
                model.addConstr(area[device] >= x[device] * devs[device]["min_area"])
                model.addConstr(area[device] <= x[device] * devs[device]["max_area"])
            
            # Correlation between PV area and peak power; cap["PV"] is only needed for calculating investment costs
            model.addConstr(cap["PV"] == area["PV"] * devs["PV"]["G_stc"] * devs["PV"]["eta"])
            
            # Correlation between STC area and peak power; cap["STC"] is only needed for calculating investment costs
            model.addConstr(cap["STC"] == area["STC"] * devs["STC"]["G_stc"] * devs["STC"]["eta"])
            
                
        
    # state of charge < storage capacity
    for device in ["TES", "CTES", "BAT", "H2S", "GS"]:
        for day in year:
            for t in time_steps:    
                model.addConstr(soc[device][day][t] <= cap[device])


    #%% INPUT / OUTPUT CONSTRAINTS
    
    for d in days:
        for t in time_steps:
            
            # Photovoltaics
            model.addConstr(power["PV"][d][t] <= param["GHI"][d][t]/1e3 * devs["PV"]["eta"] * area["PV"])
            
            # Wind turbine
            model.addConstr(power["WT"][d][t] <= devs["WT"]["norm_power"][d][t] * cap["WT"])
            
            # Hydropower
            model.addConstr(power["WAT"][d][t] <= devs["WAT"]["potential"])
            
            # Solar thermal collector
            model.addConstr(heat["STC"][d][t] <= param["GHI"][d][t]/1e3 * devs["STC"]["eta"] * area["STC"])
        
            # Electric heat pump
            model.addConstr(heat["HP"][d][t] == power["HP"][d][t] * devs["HP"]["COP"][d][t])
            
            # Electric boiler
            model.addConstr(heat["EB"][d][t] == power["EB"][d][t] * devs["EB"]["eta_th"])
            
            # Compression chiller
            model.addConstr(cool["CC"][d][t] == power["CC"][d][t] * devs["CC"]["COP"])  
    
            # Absorption chiller
            model.addConstr(cool["AC"][d][t] == heat["AC"][d][t] * devs["AC"]["eta_th"])
            
            # Gas CHP
            model.addConstr(power["CHP"][d][t] == gas["CHP"][d][t] * devs["CHP"]["eta_el"])
            model.addConstr(heat["CHP"][d][t] == gas["CHP"][d][t] * devs["CHP"]["eta_th"])
            
            # Gas boiler
            model.addConstr(heat["BOI"][d][t] == gas["BOI"][d][t] * devs["BOI"]["eta_th"])
            
            # Gas heat pump
            model.addConstr(heat["GHP"][d][t] == gas["GHP"][d][t] * devs["GHP"]["COP"])
            
            # Biomass CHP
            model.addConstr(power["BCHP"][d][t] == biom["BCHP"][d][t] * devs["BCHP"]["eta_el"])
            model.addConstr(heat["BCHP"][d][t] == biom["BCHP"][d][t] * devs["BCHP"]["eta_th"])
            
            # Biomass boiler
            model.addConstr(heat["BBOI"][d][t] == biom["BBOI"][d][t] * devs["BBOI"]["eta_th"])
                        
            # Waste CHP
            model.addConstr(power["WCHP"][d][t] == waste["WCHP"][d][t] * devs["WCHP"]["eta_el"])
            model.addConstr(heat["WCHP"][d][t] == waste["WCHP"][d][t] * devs["WCHP"]["eta_th"])
            
            # Waste boiler
            model.addConstr(heat["WBOI"][d][t] == waste["WBOI"][d][t] * devs["WBOI"]["eta_th"])
            
            # Electrolyzer
            model.addConstr(hydrogen["ELYZ"][d][t] == power["ELYZ"][d][t] * devs["ELYZ"]["eta_el"])
            
            # Fuel cell  
            model.addConstr(power["FC"][d][t] == hydrogen["FC"][d][t] * devs["FC"]["eta_el"])
            if devs["FC"]["enable_heat_diss"]:   # Heat can also be dissipated
                model.addConstr(heat["FC"][d][t] <= hydrogen["FC"][d][t] * devs["FC"]["eta_th"])
            else:   # Heat must be used
                model.addConstr(heat["FC"][d][t] == hydrogen["FC"][d][t] * devs["FC"]["eta_th"])
            
            # Sabatier reactor
            model.addConstr(gas["SAB"][d][t] == hydrogen["SAB"][d][t] * devs["SAB"]["eta"])
            
    
    #%% GLOBAL ENERGY BALANCES
    
    for d in days:
        for t in time_steps:
            
            # Heating balance
            model.addConstr(heat["STC"][d][t] + heat["HP"][d][t] + heat["EB"][d][t] + heat["CHP"][d][t] + heat["BOI"][d][t] + heat["GHP"][d][t] + heat["BCHP"][d][t] + heat["BBOI"][d][t]+ heat["WCHP"][d][t] + heat["WBOI"][d][t] + heat["FC"][d][t] == dem["heat"][d][t] + heat["AC"][d][t] + ch["TES"][d][t])
    
            # Electricity balance
            model.addConstr(power["PV"][d][t] + power["WT"][d][t] + power["WAT"][d][t] + power["CHP"][d][t] + power["BCHP"][d][t] + power["WCHP"][d][t] + power["FC"][d][t] + power["from_grid"][d][t] == dem["power"][d][t] + power["HP"][d][t] + power["EB"][d][t] + power["CC"][d][t] + power["ELYZ"][d][t] + ch["BAT"][d][t] + power["to_grid"][d][t])
    
            # Cooling balance
            model.addConstr(cool["AC"][d][t] + cool["CC"][d][t] == dem["cool"][d][t] + ch["CTES"][d][t])  
            
            # Gas balance
            model.addConstr(gas["from_grid"][d][t] + gas["SAB"][d][t] == gas["CHP"][d][t] + gas["BOI"][d][t] + gas["GHP"][d][t] + ch["GS"][d][t] + gas["to_grid"][d][t])
            
            # Hydrogen balance
            model.addConstr(hydrogen["ELYZ"][d][t] + hydrogen["import"][d][t] == dem["hydrogen"][d][t] + hydrogen["FC"][d][t] + hydrogen["SAB"][d][t] + ch["H2S"][d][t])
            
            # Biomass balance
            model.addConstr(biom["import"][d][t] == biom["BCHP"][d][t] + biom["BBOI"][d][t])
            
            # Waste balance
            model.addConstr(waste["import"][d][t] == waste["WCHP"][d][t] + waste["WBOI"][d][t])
            
    
    # Additional constraints
    #for d in days:
    #    for t in time_steps:
    #        # AC and TES can only be supplied by BOI, EB, CHP, BCHP and FC    + #heat["BCHP"][d][t] 
    #         model.addConstr(heat["BOI"][d][t] + heat["CHP"][d][t] + heat["FC"][d][t] >= heat["AC"][d][t])  # add EB
    # CTES can only be supplied by CC and AC
    #        model.addConstr(cool["AC"][d][t] + cool["CC"][d][t] >= ch["CTES"][d][t])
    
    #%% MEET PEAK DEMANDS OF UNCLUSTERED DEMANDS
    
    if param["peak_dem_met_conv"] == False:
        
        # Heating
        model.addConstr(cap["HP"] + cap["EB"]
                      + cap["CHP"] / devs["CHP"]["eta_el"] * devs["CHP"]["eta_th"] 
                      + cap["BOI"]
                      + cap["GHP"]
                      + cap["BCHP"] / devs["BCHP"]["eta_el"] * devs["BCHP"]["eta_th"]
                      + cap["BBOI"]
                      + cap["WCHP"] / devs["WCHP"]["eta_el"] * devs["WCHP"]["eta_th"]
                      + cap["WBOI"]
                      + cap["FC"] / devs["FC"]["eta_el"] * devs["FC"]["eta_th"]                    
                       >= param["peak_heat"])
                       
        # Cooling
        model.addConstr(cap["CC"] + cap["AC"] >= param["peak_cool"])
        
        # Power
        model.addConstr(cap["CHP"] + cap["BCHP"] + cap["WCHP"] + cap["FC"] + grid_limit_el >= param["peak_power"])
        
        # Hydrogen
        if param["enable_supply_limit_hydrogen"] == False:
            pass  # Hydrogen import is not limited and can always cover hydrogen demand
        else: 
            if param["supply_limit_hydrogen"] == 0:
                model.addConstr(cap["ELYZ"] >= param["peak_hydrogen"])
        
        
    else:  # With STC, PV, WIND, HYDROPOWER (WAT)
        
        # Heating
        model.addConstr(cap["STC"] + cap["HP"] + cap["EB"]
                      + cap["CHP"] / devs["CHP"]["eta_el"] * devs["CHP"]["eta_th"] 
                      + cap["BOI"]
                      + cap["GHP"]
                      + cap["BCHP"] / devs["BCHP"]["eta_el"] * devs["BCHP"]["eta_th"]
                      + cap["BBOI"]
                      + cap["WCHP"] / devs["WCHP"]["eta_el"] * devs["WCHP"]["eta_th"]
                      + cap["WBOI"]
                      + cap["FC"] / devs["FC"]["eta_el"] * devs["FC"]["eta_th"]                    
                       >= param["peak_heat"])
                       
        # Cooling
        model.addConstr(cap["CC"] + cap["AC"] >= param["peak_cool"])
        
        # Power
        model.addConstr(cap["PV"] + cap["WT"] + cap["WAT"] + cap["CHP"] + cap["BCHP"] + cap["WCHP"] + cap["FC"] + grid_limit_el >= param["peak_power"])
        
        # Hydrogen
        model.addConstr(cap["ELYZ"] + hydrogen["import"][d][t] >= param["peak_hydrogen"])
        
            
    #%% STORAGE DEVICES
        
    for device in ["TES", "CTES", "BAT", "H2S", "GS"]:
        for day in year:        
            for t in np.arange(1, len(time_steps)):
                
                # Energy balance: soc(t) = soc(t-1) + charge - discharge
                model.addConstr(soc[device][day][t] == soc[device][day][t-1] * (1-devs[device]["sto_loss"]) + ch[device][sigma[day]][t])
            
            # Transition between two consecutive days
            if day > 0:
                model.addConstr(soc[device][day][0] == soc[device][day-1][len(time_steps)-1] * (1-devs[device]["sto_loss"]) + ch[device][sigma[day]][0])
                
        # Cyclic year condition
        model.addConstr(soc[device][0][0] ==  soc[device][len(year)-1][len(time_steps)-1] * (1-devs[device]["sto_loss"]) + ch[device][sigma[0]][0])


    #%% SUM UP RESULTS
    
    ### Total energy import/feed-in ###
    # Total amount of gas taken from and to grid
    model.addConstr(from_gas_grid_total == sum(sum(gas["from_grid"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))
    model.addConstr(to_gas_grid_total == sum(sum(gas["to_grid"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))
  
    # Total electric energy from and to grid
    model.addConstr(from_el_grid_total == sum(sum(power["from_grid"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))
    model.addConstr(to_el_grid_total == sum(sum(power["to_grid"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))
    
    # Total amount of biomass imported
    model.addConstr(biom_import_total == sum(sum(biom["import"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))
    
    # Total amount of waste imported
    model.addConstr(waste_import_total == sum(sum(waste["import"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))
    
    # Total amount of hydrogen imported
    model.addConstr(hydrogen_import_total == sum(sum(hydrogen["import"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))
    

    ### Costs ###
    # Costs/revenues for electricity
    model.addConstr(supply_costs_el  == from_el_grid_total  * param["price_supply_el"])
    model.addConstr(cap_costs_el     == grid_limit_el       * param["price_cap_el"])
    model.addConstr(rev_feed_in_el   == to_el_grid_total    * param["revenue_feed_in_el"])
    
    # Costs/revenues for natural gas
    model.addConstr(supply_costs_gas == from_gas_grid_total * param["price_supply_gas"])
    model.addConstr(cap_costs_gas    == grid_limit_gas      * param["price_cap_gas"])
    model.addConstr(rev_feed_in_gas  == to_gas_grid_total   * param["revenue_feed_in_gas"])
    
    # Costs for biomass, waste and hydrogen
    model.addConstr(supply_costs_biom     == biom_import_total     * param["price_biomass"])
    model.addConstr(supply_costs_waste    == waste_import_total    * param["price_waste"])
    model.addConstr(supply_costs_hydrogen == hydrogen_import_total * param["price_hydrogen"])
    
    
    ### Supply limitations ###
    # Forbid/allow feed-in (user input)
    if param["enable_feed_in_el"] != True:
        model.addConstr(to_el_grid_total == 0)
    if param["enable_feed_in_gas"] != True:
        model.addConstr(to_gas_grid_total == 0)
    
    # Limitation of electricity supply (user input)
    if param["enable_cap_limit_el"] == True:
        model.addConstr(grid_limit_el <= param["cap_limit_el"])
    if param["enable_supply_limit_el"] == True:
        model.addConstr(from_el_grid_total <= param["supply_limit_el"])
    
    # Limitation of gas supply (user input)
    if param["enable_cap_limit_gas"] == True:
        model.addConstr(grid_limit_gas <= param["cap_limit_gas"])
    if param["enable_supply_limit_gas"] == True:
        model.addConstr(from_gas_grid_total <= param["supply_limit_gas"])
        
    # Limitation of biomass, waste and hydrogen supply (user input)
    if param["enable_supply_limit_biom"] == True:
        model.addConstr(biom_import_total <= param["supply_limit_biom"])
    if param["enable_supply_limit_waste"] == True:    
        model.addConstr(waste_import_total <= param["supply_limit_waste"])
    if param["enable_supply_limit_hydrogen"] == True:    
        model.addConstr(hydrogen_import_total <= param["supply_limit_hydrogen"])
        
    # Total investment costs
    for device in all_devs:
        model.addConstr(inv[device] == devs[device]["inv_var"] * cap[device])  
        
    # Annual investment costs
    for device in all_devs:
        model.addConstr(c_inv[device] == inv[device] * devs[device]["ann_factor"])
        
    # Operation and maintenance costs
    for device in all_devs:       
        model.addConstr(c_om[device] == devs[device]["cost_om"] * inv[device])
        
    # Total annual costs
    for device in all_devs:
        model.addConstr(c_total[device] == c_inv[device] + c_om[device])
        

    #%% OBJECTIVE FUNCTIONS
    # Total annualized costs
    model.addConstr(obj["tac"] == sum(c_total[dev] for dev in all_devs) # annualized investments
                                + supply_costs_gas + cap_costs_gas # gas costs
                                + supply_costs_el + cap_costs_el # electricity costs
                                - rev_feed_in_el - rev_feed_in_gas # revenues                                
                                + supply_costs_biom # biomass
                                + supply_costs_waste # waste
                                + supply_costs_hydrogen
                                + (from_gas_grid_total * param["co2_gas"] + biom_import_total * param["co2_biom"] + waste_import_total * param["co2_waste"]) * param["co2_tax"]) # CO2 tax
                                
    
    # Annual CO2 emissions: Implicit emissions by power supply from national grid is penalized, feed-in is ignored
    model.addConstr(obj["co2"] == from_el_grid_total * param["co2_el_grid"]       
                                      + from_gas_grid_total * param["co2_gas"] 
                                      + biom_import_total * param["co2_biom"]
                                      + waste_import_total * param["co2_waste"]
                                      + hydrogen_import_total * param["co2_hydrogen"]
                                      - to_el_grid_total * param["co2_el_feed_in"]
                                      - to_gas_grid_total * param["co2_gas_feed_in"])

                                      
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Set model parameters and execute calculation

    print("Precalculation and model set up done in %f seconds."  % (time.time() - start_time))

    # Set solver parameters
    model.Params.MIPGap = 0.02  # ---,         gap for branch-and-bound algorithm
    #model.Params.Heuristics = 0.15
    model.Params.method     = 2                         # ---,         -1: default, 0: primal simplex, 1: dual simplex, 2: barrier, etc.

    # Execute calculation
    start_time = time.time()
    model.optimize()
    print("Optimization done. (%f seconds.)" % (time.time() - start_time))

    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Check and save results

    # Check if optimal solution was found
    if model.Status in (3, 4) or model.SolCount == 0:  # "INFEASIBLE" or "INF_OR_UNBD"
        flags["optimal_solution_found"] = False
        print("Optimization: No feasible solution found.")
        try:
            #pass
            print("Try to calculate IIS.")
            model.computeIIS()
            model.write("model.ilp")
            print("IIS was calculated and saved as model.ilp")
        except:
            print("Could not calculate IIS.")
        return {}, flags

    else:
        
        #model.write("model.sol")
        #model.write("model.lp")
        
        result_dict["tac"] = int(obj["tac"].X) # EUR/a    
        result_dict["co2"] = int(obj["co2"].X/1000) # t/a

        for k in cap.keys():
            result_dict[k] = {"cap": round(cap[k].X, 1)}

        result_dict["total_inv_cost"] = int(sum(inv[k].X for k in cap.keys()))
        result_dict["total_ann_inv_cost"] = int(sum(c_inv[k].X for k in cap.keys()))
        result_dict["total_om_cost"] = int(sum(c_om[k].X  for k in cap.keys()))
        result_dict["from_el_grid_total"] = int(from_el_grid_total.X / 1000)  # MWh
        result_dict["to_el_grid_total"] = int(to_el_grid_total.X / 1000)  # MWh
        result_dict["from_gas_grid_total"] = int(from_gas_grid_total.X / 1000)  # MWh
        result_dict["to_gas_grid_total"] = int(to_gas_grid_total.X / 1000)  # MWh
        result_dict["biom_import_total"] = int(biom_import_total.X / 1000)  # MWh
        result_dict["waste_import_total"] = int(waste_import_total.X / 1000)  # MWh
        result_dict["hydrogen_import_total"] = int(hydrogen_import_total.X / 1000)  # MWh
        
        # CO2 emissions
        result_dict["co2_onsite_emissions"] = int((from_gas_grid_total.X * param["co2_gas"] + biom_import_total.X * param["co2_biom"] + waste_import_total.X * param["co2_waste"])/1000)
        result_dict["co2_global_emissions"] = int(result_dict["co2"]/1000)
        result_dict["co2_credit_feedin"] = int((to_el_grid_total.X * param["co2_el_feed_in"] + to_gas_grid_total.X * param["co2_gas_feed_in"])/1000)
        result_dict["co2_tax_total"] = int(result_dict["co2_onsite_emissions"] * param["co2_tax"] * 1000)  # EUR, only gas, biomass and waste.
        
        # Calculate maximum grid flows (electricity and gas)
        for k in ["from_grid", "to_grid"]:
            result_dict["max_el_" + k] = 0
            for d in days:
                for t in time_steps:
                    if power[k][d][t].X > result_dict["max_el_" + k]:
                        result_dict["max_el_" + k] = power[k][d][t].X
            result_dict["max_el_" + k] = int(result_dict["max_el_" + k])
        
        for k in ["from_grid", "to_grid"]:
            result_dict["max_gas_" + k] = 0
            for d in days:
                for t in time_steps:
                    if gas[k][d][t].X > result_dict["max_gas_" + k]:
                        result_dict["max_gas_" + k] = gas[k][d][t].X            
            result_dict["max_gas_" + k] = int(result_dict["max_gas_" + k])
            
        result_dict["max_biom"] = 0
        for d in days:
            for t in time_steps:
                if biom["import"][d][t].X > result_dict["max_biom"]:
                    result_dict["max_biom"] = biom["import"][d][t].X            
        result_dict["max_biom"] = int(result_dict["max_biom"])
         
        result_dict["max_waste"] = 0
        for d in days:
            for t in time_steps:
                if waste["import"][d][t].X > result_dict["max_waste"]:
                    result_dict["max_waste"] = waste["import"][d][t].X            
        result_dict["max_waste"] = int(result_dict["max_waste"])
        
        result_dict["max_hydrogen"] = 0
        for d in days:
            for t in time_steps:
                if hydrogen["import"][d][t].X > result_dict["max_hydrogen"]:
                    result_dict["max_hydrogen"] = hydrogen["import"][d][t].X            
        result_dict["max_hydrogen"] = int(result_dict["max_hydrogen"])

        result_dict["supply_costs_el"] = int(supply_costs_el.X)
        result_dict["cap_costs_el"] = int(cap_costs_el.X)
        result_dict["total_el_costs"] = int(supply_costs_el.X + cap_costs_el.X)
        result_dict["rev_feed_in_el"] = int(rev_feed_in_el.X)
        
        result_dict["supply_costs_gas"] = int(supply_costs_gas.X)
        result_dict["cap_costs_gas"] = int(cap_costs_gas.X)
        result_dict["total_gas_costs"] = int(supply_costs_gas.X + cap_costs_gas.X)
        result_dict["rev_feed_in_gas"] = int(rev_feed_in_gas.X)
        
        result_dict["supply_costs_biom"] = int(supply_costs_biom.X)
        result_dict["supply_costs_waste"] = int(supply_costs_waste.X)
        result_dict["supply_costs_hydrogen"] = int(supply_costs_hydrogen.X)
        
        # Prepare time series of renewable curtailment
        power["PV_curtail"] = {}
        power["WT_curtail"] = {}
        power["WAT_curtail"] = {}
        heat["STC_curtail"] = {}
        for d in days:
            power["PV_curtail"][d] = {}
            power["WT_curtail"][d] = {}
            power["WAT_curtail"][d] = {}
            heat["STC_curtail"][d] = {}
            for t in time_steps:
                power["PV_curtail"][d][t] = param["GHI"][d][t]/1e3 * devs["PV"]["eta"] * area["PV"].X - power["PV"][d][t].X
                
                power["WT_curtail"][d][t] = devs["WT"]["norm_power"][d][t] * cap["WT"].X - power["WT"][d][t].X
                
                power["WAT_curtail"][d][t] = np.min([cap["WAT"].X, devs["WAT"]["potential"]]) - power["WAT"][d][t].X
                
                
                heat["STC_curtail"][d][t] = param["GHI"][d][t]/1e3 * devs["STC"]["eta"] * area["STC"].X - heat["STC"][d][t].X
                
        
        result_dict["PV"]["curtailed"] = int((sum(sum(power["PV_curtail"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))/1000)
        
        result_dict["STC"]["curtailed"] = int((sum(sum(heat["STC_curtail"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))/1000)

        result_dict["WT"]["curtailed"] = int((sum(sum(power["WT_curtail"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))/1000)
        
        result_dict["WAT"]["curtailed"] = int((sum(sum(power["WAT_curtail"][d][t] for t in time_steps) * param["day_weights"][d] for d in days))/1000)

        
        # Calculate generation
        eps = 0.01
        for k in ["STC", "HP", "EB", "BOI", "GHP", "BBOI", "WBOI"]:
            result_dict[k]["gen_kWh"] = sum(sum(heat[k][d][t].X for t in time_steps) * param["day_weights"][d] for d in days)  # in kWh
            result_dict[k]["gen"] = int(sum(sum(heat[k][d][t].X for t in time_steps) * param["day_weights"][d] for d in days)/1000)  # in MWh
            
        for k in ["CC", "AC"]:
            result_dict[k]["gen_kWh"] = sum(sum(cool[k][d][t].X for t in time_steps) * param["day_weights"][d] for d in days)  # in kWh
            result_dict[k]["gen"] = int(sum(sum(cool[k][d][t].X for t in time_steps) * param["day_weights"][d] for d in days)/1000)  # in MWh
        
        for k in ["PV", "WT", "WAT", "CHP", "BCHP", "WCHP", "ELYZ", "FC"]:  # WAT
            result_dict[k]["gen_kWh"] = sum(sum(power[k][d][t].X for t in time_steps) * param["day_weights"][d] for d in days)  # in kWh
            result_dict[k]["gen"] = int(sum(sum(power[k][d][t].X for t in time_steps) * param["day_weights"][d] for d in days)/1000)  # in MWh
        
        # Calculate hydrogen generation for ELYZ
        result_dict["ELYZ"]["gen_H2"] = int(sum(sum(power["ELYZ"][d][t].X * devs["ELYZ"]["eta_el"]for t in time_steps) * param["day_weights"][d] for d in days)/1000)  # in MWh
            
        for k in ["SAB"]:
            result_dict[k]["gen_kWh"] = sum(sum(gas[k][d][t].X for t in time_steps) * param["day_weights"][d] for d in days)  # in kWh
            result_dict[k]["gen"] = int(sum(sum(gas[k][d][t].X for t in time_steps) * param["day_weights"][d] for d in days)/1000)  # in MWh
            
        # Calculate full load hours
        for k in ["PV", "WT", "WAT", "STC", "HP", "EB", "CC", "AC", "CHP", "BOI", "GHP", "BCHP", "BBOI", "WCHP", "WBOI", "ELYZ", "FC", "SAB"]:
            if cap[k].X > eps:
                result_dict[k]["hrs"] = int(result_dict[k]["gen_kWh"] / cap[k].X)
            else:
                result_dict[k]["hrs"] = 0
                
                
        # Select technologies that are installed (to list only these in results)
        for k in all_devs:
            if cap[k].X > eps:
                result_dict[k]["inst"] = True
            else:
                result_dict[k]["inst"] = False
        result_dict["PV_or_STC_inst"] = (result_dict["PV"]["inst"] or result_dict["STC"]["inst"])
        result_dict["right_gen_tech_inst"] =(result_dict["WT"]["inst"] or result_dict["WAT"]["inst"] or result_dict["ELYZ"]["inst"] or result_dict["FC"]["inst"] or result_dict["SAB"]["inst"])
        result_dict["storage_inst"] = (result_dict["TES"]["inst"] or result_dict["CTES"]["inst"] or result_dict["GS"]["inst"] or result_dict["BAT"]["inst"] or result_dict["H2S"]["inst"])
        
        
        # Area of PV and STC
        result_dict["PV"]["area"] = int(area["PV"].X)
        result_dict["STC"]["area"] = int(area["STC"].X)
        
        # Calculate charge cycles of storages
        for k in ["TES", "CTES", "BAT", "H2S", "GS"]:
            if cap[k].X > eps:
                result_dict[k]["chc"] = int(sum(sum(abs(ch[k][d][t].X)/2 for t in time_steps) * param["day_weights"][d] for d in days) / cap[k].X)
            else:
                result_dict[k]["chc"] = 0
        
        # Calculate volume of thermal storages
        for k in ["TES", "CTES"]:
            result_dict[k]["vol"] = round(cap[k].X / (param["c_w"] * param["rho_w"] * devs[k]["delta_T"]) * 3600, 1)
            
            
        # Calculate emissions
        result_dict["total_co2_el"] = int(from_el_grid_total.X * param["co2_el_grid"]/1000) # t/a
        result_dict["total_co2_el_feed_in"] = int(to_el_grid_total.X * param["co2_el_feed_in"]/1000) # t/a
        if result_dict["total_co2_el_feed_in"] > 0:
            result_dict["total_co2_el_feed_in_larger_zero"] = True
        result_dict["total_co2_gas"] = int(from_gas_grid_total.X * param["co2_gas"]/1000) # t/a 
        result_dict["total_co2_gas_feed_in"] = int(to_gas_grid_total.X * param["co2_gas_feed_in"]/1000) # t/a 
        if result_dict["total_co2_gas_feed_in"] > 0:
            result_dict["total_co2_gas_feed_in_larger_zero"] = True
        result_dict["total_co2_biom"] = int(biom_import_total.X * param["co2_biom"]/1000) # t/a
        result_dict["total_co2_waste"] = int(waste_import_total.X * param["co2_waste"]/1000) # t/a
        result_dict["total_co2_hydrogen"] = int(hydrogen_import_total.X * param["co2_hydrogen"]/1000) # t/a
            
        # Calculate share of renewables     #add WAT
        result_dict["share_renew"] = round((result_dict["PV"]["gen_kWh"] + result_dict["WT"]["gen_kWh"] + result_dict["WAT"]["gen_kWh"] + result_dict["STC"]["gen_kWh"])/(result_dict["PV"]["gen_kWh"] + result_dict["WT"]["gen_kWh"] + result_dict["WAT"]["gen_kWh"] + result_dict["STC"]["gen_kWh"] + from_el_grid_total.X + from_gas_grid_total.X + biom_import_total.X + waste_import_total.X + hydrogen_import_total.X)*100, 1)  # 
        
        # Calculate relative savings compared to reference scenario
        if not result_dict["ref"]["tac"] == 0:
            if result_dict["tac"] <= result_dict["ref"]["tac"]:
                result_dict["ref"]["tac_sav"] = round((1-(result_dict["tac"]/result_dict["ref"]["tac"])) * 100, 1)
                result_dict["ref"]["tac_sav_pos"] = True
            else:
                result_dict["ref"]["tac_sav"] = round(((result_dict["tac"]/result_dict["ref"]["tac"])-1) * 100, 1)
                result_dict["ref"]["tac_sav_pos"] = False
        else: 
            result_dict["ref"]["tac_sav"] = 0
            
        if not result_dict["ref"]["co2"] == 0:
            if result_dict["co2"] <= result_dict["ref"]["co2"]:
                result_dict["ref"]["co2_sav"] = round((1-(result_dict["co2"]/result_dict["ref"]["co2"])) * 100, 1)
                result_dict["ref"]["co2_sav_pos"] = True
            else:
                result_dict["ref"]["co2_sav"] = round(((result_dict["co2"]/result_dict["ref"]["co2"])-1) * 100, 1)
                result_dict["ref"]["co2_sav_pos"] = False
        else: 
            result_dict["ref"]["co2_sav"] = 0
        
        result_dict = create_excel_file(heat, cool, power, gas, biom, waste, hydrogen, ch, soc, dem, param, time_steps, days, result_dict)
        
        return result_dict, flags

        
def create_excel_file(heat, cool, power, gas, biom, waste, hydrogen, ch, soc, dem, param, time_steps, days, result_dict):

    ### GENERATION ###
    techs = {"power_PV": "PV power (kW_el)",
             "power_PV_curtail": "PV power curtailed (kW_el)",
             "power_WT": "Wind power (kW_el)",
             "power_WT_curtail": "Wind power curtailed (kW_el)",
             "power_WAT": "Hydropower (kW_el)",
             "power_WAT_curtail": "Hydropower curtailed (kW_el)",
             "heat_STC": "Thermal output solar thermal collector (kW_th)",
             "heat_STC_curtail": "Solar thermal collector curtailment (kW_th)",
             
             "heat_HP": "Thermal output electric heat pump (kW_th)",
             "power_HP": "Electric power heat pump (kW_el)",
             "heat_EB": "Thermal output electric boiler (kW_th)",
             "power_EB": "Electric power electric boiler (kW_el)",
             "cool_CC": "Cooling power compression chiller (kW_th)",
             "power_CC": "Electric power compression chiller (kW_el)",
             "cool_AC": "Cooling power absorption chiller (kW_th)", 
             "heat_AC": "Heat demand absorption chiller (kW_th)", 
             
             "power_CHP": "CHP unit power (kW_el)",
             "heat_CHP": "Thermal output CHP unit (kW_el)",
             "heat_BOI": "Heat gas boiler (kW_th)",
             "gas_BOI": "Gas demand gas boiler (kW)",
             "heat_GHP": "Heat output gas heat pump (kW_th)",
             "gas_GHP": "Gas demand gas heat pump (kW)",
                
             "power_BCHP": "Biomass CHP power (kW_el)",
             "heat_BCHP": "Heat output biomass CHP (kW_th)",
             "biom_BCHP": "Biomass demand biomass CHP (kW)",
             "heat_BBOI": "Heat output biomass boiler (kW_th)",
             "biom_BBOI": "Biomass demand biomass boiler (kW_th)",
             "power_WCHP": "Waste CHP power (kW_el)",
             "heat_WCHP": "Heat output waste CHP (kW_th)",
             "waste_WCHP": "Waste demand waste CHP (kW)",
             "heat_WBOI": "Heat output waste boiler (kW_th)",
             "waste_WBOI": "Waste demand waste boiler (kW)",
             
             "power_ELYZ": "Electrolyzer (kW_el)",
             "hydrogen_ELYZ": "Hydrogen generation electrolyzer (kW)",
             "power_FC": "Fuel cell (kW_el)",
             "heat_FC": "Fuel cell (kW_th)",
             "hydrogen_FC": "Hydrogen demand fuel cell (kW)",
             "hydrogen_SAB": "Hydrogen demand sabatier reactor (kW)",
             "gas_SAB": "Gas output sabatier reactor (kW)",
             
             "ch_TES": "Charging heat storage (kW_th)",
             "ch_CTES": "Charging cold storage (kW_th)",
             "ch_BAT": "Charging battery (kW_el)",
             "ch_GS": "Charging gas storage (kW)",
             "ch_H2S": "Charging hydrogen storage (kW)",
             
             "dem_heat": "Heating demand (kW)",
             "dem_cool": "Cooling demand (kW)",
             "dem_power": "Electricity demand (kW)",
             "dem_hydrogen": "Hydrogen demand (kW)",
             
             "biom_import": "Import biomass (kW)",
             "waste_import": "Import waste (kW)",
             "hydrogen_import": "Import hydrogen (kW)",
             "power_to_grid": "Electricity feed-in (kW)",
             "power_from_grid": "Electricity import (kW)",
             }
             
    socs = {"soc_TES": "State of charge heat storage (kWh_th)",
            "soc_CTES": "State of charge cold storage (kWh_th)",
            "soc_BAT": "State of charge battery (kWh_el)",
            "soc_GS": "State of charge gas storage (kWh)",
            "soc_H2S": "State of charge hydrogen storage (kWh)",
            }

    
             
             
    ### REWRITE DESIGN DAYS IN FULL YEAR ###
    
    # Arrange full time series with 8760 steps
    full = {}
    for item in techs.keys():
        full[item] = np.zeros(8760)   
    # Get list of days used as type days
    z = param["day_matrix"]
    typedays = []
    for d in range(365):
        if any(z[d]):
            typedays.append(d)
    # Arrange time series
    for d in range(365):
        match = np.where(z[:,d] == 1)[0][0]
        typeday = np.where(typedays == match)[0][0]        
        for item in techs.keys():
            m, tech = item.split("_", 1)
            if not m == "soc":
                if m == "power":
                    m_arr = power
                elif m == "heat":
                    m_arr = heat
                elif m == "cool":
                    m_arr = cool
                elif m == "hydrogen":
                    m_arr = hydrogen
                elif m == "gas":
                    m_arr = gas
                elif m == "biom":
                    m_arr = biom
                elif m == "waste":
                    m_arr = waste
                elif m == "ch":
                    m_arr = ch
                elif m == "dem":
                    m_arr = dem
                for t in range(24):    
                    if m == "dem" or tech == "PV_curtail" or tech == "STC_curtail" or tech == "WT_curtail" or tech == "WAT_curtail":
                        full[item][24*d+t] = m_arr[tech][typeday][t]
                    else:
                        full[item][24*d+t] = m_arr[tech][typeday][t].X
                    # print("full["+item+"][" + str(24*d+t) + "] = m_arr["+tech+"]["+str(typeday)+"]["+str(t)+"].X)")

    for item in socs.keys():
        m, tech = item.split("_", 1)
        if m == "soc":
            full[item] = np.zeros(8760)   
            for d in range(365):
                for t in range(24):
                    full[item][24*d+t] = soc[tech][d][t].X
                    
    ### CALC MONTHLY VALS ###
    
    month_tuple = ("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
    days_sum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    # Calc ambient heat
    full["amb_heat_HP"] = full["heat_HP"] - full["power_HP"]
    
    
    # Calc mean COP
    result_dict["mean_COP_HP"] = round(np.sum(full["heat_HP"]) / np.sum(full["power_HP"]),2)
    
    monthly_val = {}
    year_peak = {}
    year_sum = {}
    for m in ["power_PV", "power_WT", "power_WAT", "heat_STC", "heat_HP", "amb_heat_HP"]:
        monthly_val[m] = {}
        year_peak[m] = int(np.max(full[m]))
        year_sum[m] = int(np.sum(full[m]) / 1000)
        for month in range(12):
            monthly_val[m][month_tuple[month]] = sum(full[m][t] for t in range(days_sum[month]*24, days_sum[month+1]*24)) / 1000
    
    result_dict["monthly_val"] = monthly_val
    result_dict["year_peak"].update(year_peak) 
    result_dict["year_sum"].update(year_sum)
                    
                    
    ### WRITE EXCEL FILE ###
                    
    import xlsxwriter
    import os
    
    file_path = settings.EXCEL_PATH
    if not os.path.exists(settings.MEDIA_ROOT):
        os.makedirs(settings.MEDIA_ROOT)
    
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet("Time series")
        
    boldcenter = workbook.add_format({"bold": 1})
    
    col=0
    for k in techs.keys():
        worksheet.write(0,col,techs[k], boldcenter)
        for t in range(8760):        
            worksheet.write(t+1,col,full[k][t])
        col+=1
        
    for k in socs.keys():
        worksheet.write(0,col,socs[k], boldcenter)
        for t in range(8760):        
            worksheet.write(t+1,col,full[k][t])
        col+=1
        
    # Increase column width
    worksheet.set_column(0,100,20) 
    
    for k in ["heat", "cool", "power", "hydrogen"]:
        worksheet = workbook.add_worksheet("Clustered " + str(k) + " demands")
        for d in days: 
            for t in time_steps:
                worksheet.write(t,d,dem[k][d][t])
            
    workbook.close()
    
    return result_dict