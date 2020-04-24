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

import numpy as np
import math
import optim_app.help_functions.clustering_medoid as clustering
import time

from optim_app.models import Location, City, Country


def load_params(all_models, flags):

    result_dict = {}

    param = {}
    param_uncl = {}  # unclustered time series for weather data

    path_input_data = "optim_app/help_functions/input_data/"
    
    
    ################################################################
    # GENERAL PARAMETERS
    
    param["c_w"] = 4.18  # kJ/(kgK)
    param["rho_w"] = 1000  # kg/m3
    
    ################################################################
    # LOAD WEATHER DATA

    city = str(all_models["Location"].city)
    country = str(all_models["Location"].country)

    file_name = str(all_models["Location"].city.file)
    file = open(path_input_data + "/weather_files/" + file_name, "rb")
    T_air, GHI, wind_speed = np.loadtxt(file, delimiter=",", skiprows=8, usecols=[6,13,21], unpack=True)
    
    param_uncl["T_air"] = T_air
    param_uncl["GHI"] = GHI
    param_uncl["wind_speed"] = wind_speed    
    
    name_dict = {"AirTemperature": "T_air", 
                 "SolarIrradiance": "GHI", 
                 "WindSpeed": "wind_speed"
                 }
    
    used_time_series_of_location = False
    for k in name_dict.keys():    
        if all_models[k].upload:  # overwrite with uploaded time series, if it is valid
            try:
                param_uncl[name_dict[k]] = np.loadtxt(all_models[k].file_time_series)
                flags[k] = (len(param_uncl[name_dict[k]]) == 8760)
                print("Time series for " + name_dict[k] + ": " + str(param_uncl[name_dict[k]]))
            except:
                flags[k] = False
        else:
            used_time_series_of_location = True
            
    if used_time_series_of_location:
        result_dict["country"] = all_models["Location"].country.name
        result_dict["city"] = all_models["Location"].city.name
    else:
        result_dict["country"] = "---"
        result_dict["city"] = "---"
    
    ################################################################
    # LOAD DEMANDS

    dem_uncl = {}
    
    name_dict = {"HeatDemand": "heat", 
                 "CoolDemand": "cool", 
                 "PowerDemand": "power", 
                 "H2Demand": "hydrogen"
                 }
    
    example_file_dict = {"HeatDemand": "heating_demand.txt", 
                         "CoolDemand": "cooling_demand.txt", 
                         "PowerDemand": "electricity_demand.txt", 
                         "H2Demand": "hydrogen_demand.txt"
                         }
    
    path_example_data = "optim_app/static/" 
    
    for k in name_dict.keys():  
        if (all_models[k].use_example or all_models[k].is_constant or all_models[k].upload) == False:
            flags[k+"_no_option"] = False
            
        elif all_models[k].use_example:
            dem_uncl[name_dict[k]] = np.loadtxt(path_example_data+example_file_dict[k])  # kW
            
        elif all_models[k].is_constant:
            dem_uncl[name_dict[k]] = np.ones(8760) * all_models[k].const_demand  # kW

        elif all_models[k].upload:
            try:
                dem_uncl[name_dict[k]] = np.loadtxt(all_models[k].file_time_series)  # kW
                flags[k+"_invalid_file"] = (len(dem_uncl[name_dict[k]]) == 8760)
            except:
                flags[k+"_invalid_file"] = False

        else:
            flags[k] = False
        #print(dem_uncl[name_dict[k]])
        
    if not all(val == True for val in flags.values()):
        return {}, {}, {}, {}, flags
    
    for k in name_dict.keys():  
        param["peak_"+name_dict[k]] = np.max(dem_uncl[name_dict[k]])

        
    ################################################################
    # DESIGN DAY CLUSTERING
    
    param["n_clusters"] = int(all_models["General"].design_days)  # number of design days
    
    # Collect the time series to be clustered   
    time_series = [dem_uncl["heat"], dem_uncl["cool"], dem_uncl["power"], dem_uncl["hydrogen"], param_uncl["T_air"], param_uncl["GHI"], param_uncl["wind_speed"]]
    # Only building demands and weather data are clustered using k-medoids algorithm; secondary time series are clustered manually according to k-medoids result
    inputs_clustering = np.array(time_series)                        
    # Execute k-medoids algorithm
    print("Cluster design days...")  
    start = time.time()    
    (clustered_series, nc, z) = clustering.cluster(inputs_clustering, 
                                     param["n_clusters"],
                                     norm = 2,
                                     mip_gap = 0.02,
                                     )
    print("Design day clustering finished. (" + str(time.time()-start) + ")\n")
    
    dem = {}
    dem["heat"] = clustered_series[0]
    dem["cool"] = clustered_series[1]
    dem["power"] = clustered_series[2]
    dem["hydrogen"] = clustered_series[3]
    param["T_air"] = clustered_series[4]
    param["GHI"] = clustered_series[5]
    param["wind_speed"] = clustered_series[6]
    
    # Save number of design days and design-day matrix
    param["day_weights"] = nc
    param["day_matrix"] = z
    
    # Get sigma-function: for each day of the year, find the corresponding design day
    # Get list of days which are used as design days
    typedays = np.zeros(param["n_clusters"], dtype = np.int32)
    n = 0
    for d in range(365):
        if any(z[d]):
            typedays[n] = d
            n += 1
    # Assign each day of the year to its design day
    sigma = np.zeros(365, dtype = np.int32)
    for day in range(len(sigma)):
        d = np.where(z[:,day] == 1 )[0][0]
        sigma[day] = np.where(typedays == d)[0][0]
    param["sigma"] = sigma
    
    # Cluster secondary time series
    #for k in ["T_air", "GHI", "wind_speed"]:
    #    series_clustered = np.zeros((param["n_clusters"], 24))
    #    for d in range(param["n_clusters"]):
    #        for t in range(24):
    #            series_clustered[d][t] = param_uncl[k][24*typedays[d]+t]
        # Replace original time series with the clustered one
    #    param[k] = series_clustered  
        
        
    ################################################################
    # LOAD TECHNICAL PARAMETERS
    
    
    devs = {}
    
    # Photovoltaics
    devs["PV"] = {
        "feasible":  all_models["Photovoltaic"].enabled,
        "eta":       all_models["Photovoltaic"].eta / 100,
        "life_time": all_models["Photovoltaic"].life_time,
        "inv_var":   all_models["Photovoltaic"].inv_var,   
        "cost_om":   all_models["Photovoltaic"].cost_om / 100,
        "max_area":  all_models["Photovoltaic"].max_area,
        "min_area":  all_models["Photovoltaic"].min_area,
        # For correlation between area and peak power:
        "G_stc": 1, # kW/m^2,  solar radiation under standard test conditions (STC)                      
        }
        
    # Wind turbine
    devs["WT"] = {
        "feasible":  all_models["WindTurbine"].enabled,
        "inv_var":   all_models["WindTurbine"].inv_var,
        "life_time": all_models["WindTurbine"].life_time,
        "cost_om":   all_models["WindTurbine"].cost_om / 100,
        "min_cap":   all_models["WindTurbine"].min_cap,
        "max_cap":   all_models["WindTurbine"].max_cap,
        "h_coeff":   all_models["WindTurbine"].h_coeff,  # hellmann_coeff
        "hub_h":     all_models["WindTurbine"].hub_h,
        "ref_h":     all_models["WindTurbine"].ref_h,
        }        
    devs["WT"]["norm_power"] = calc_WT_power(devs, param) # relative power between 0 and 1
                  
    # Hydropower
    devs["WAT"] = {
        "feasible":  all_models["Hydropower"].enabled,
        "inv_var":   all_models["Hydropower"].inv_var,
        "life_time": all_models["Hydropower"].life_time,
        "cost_om":   all_models["Hydropower"].cost_om / 100,
        "min_cap":   all_models["Hydropower"].min_cap,
        "max_cap":   all_models["Hydropower"].max_cap,
        "potential": all_models["Hydropower"].potential,
        }
    
    # Solar thermal collector
    devs["STC"] = {
        "feasible":  all_models["SolarThermalCollector"].enabled,
        "eta":       all_models["SolarThermalCollector"].eta / 100,
        "inv_var":   all_models["SolarThermalCollector"].inv_var,
        "life_time": all_models["SolarThermalCollector"].life_time,
        "cost_om":   all_models["SolarThermalCollector"].cost_om / 100,
        "max_area":  all_models["SolarThermalCollector"].max_area,
        "min_area":  all_models["SolarThermalCollector"].min_area,
        # For correlation between area and peak power:
        "G_stc": 1, # kW/m^2,  solar radiation under standard test conditions (STC)                   
        }
    
    
    ### Natural gas ###
    
    # CHP
    devs["CHP"] = {
        "feasible":  all_models["CHP"].enabled,
        "inv_var":   all_models["CHP"].inv_var,
        "eta_el":    all_models["CHP"].eta_el / 100,
        "eta_th":    all_models["CHP"].eta_th / 100,
        "life_time": all_models["CHP"].life_time,
        "cost_om":   all_models["CHP"].cost_om / 100,
        "min_cap":   all_models["CHP"].min_cap,
        "max_cap":   all_models["CHP"].max_cap,
    }
    
    # Gas boiler
    devs["BOI"] = {
        "feasible":  all_models["GasBoiler"].enabled,
        "inv_var":   all_models["GasBoiler"].inv_var,
        "eta_th":    all_models["GasBoiler"].eta_th / 100,
        "life_time": all_models["GasBoiler"].life_time,
        "cost_om":   all_models["GasBoiler"].cost_om / 100,
        "min_cap":   all_models["GasBoiler"].min_cap,
        "max_cap":   all_models["GasBoiler"].max_cap,
    }

    # Gas heat pump
    devs["GHP"] = {
        "feasible":  all_models["GasHeatPump"].enabled,
        "inv_var":   all_models["GasHeatPump"].inv_var,
        "COP":       all_models["GasHeatPump"].COP,
        "life_time": all_models["GasHeatPump"].life_time,
        "cost_om":   all_models["GasHeatPump"].cost_om / 100,
        "min_cap":   all_models["GasHeatPump"].min_cap,
        "max_cap":   all_models["GasHeatPump"].max_cap,
    }
    
    
    ### Heating and cooling ###
    
    # Heat pump (depending on investment and COP, it can be air source or ground source heat pump)
    devs["HP"] = {
        "feasible": all_models["HeatPump"].enabled,
        "inv_var": all_models["HeatPump"].inv_var,
        "life_time": all_models["HeatPump"].life_time,
        "cost_om": all_models["HeatPump"].cost_om / 100,
        "min_cap": all_models["HeatPump"].min_cap,
        "max_cap": all_models["HeatPump"].max_cap,
    }
    # COP assignment
    if all_models["HeatPump"].enabled:
        if (not all_models["HeatPump"].COP_is_const) and (not all_models["HeatPump"].is_ASHP) and (not all_models["HeatPump"].COP_is_time_series):
            flags["HeatPump_no_COP_option_selected"] = False
        elif all_models["HeatPump"].COP_is_const:
            devs["HP"]["COP"] = np.ones((param["n_clusters"], 24)) * all_models["HeatPump"].COP_const
        elif all_models["HeatPump"].is_ASHP:
            COP = np.ones((param["n_clusters"], 24))
            eta_carnot = all_models["HeatPump"].ASHP_carnot_eff / 100
            supply_temp = all_models["HeatPump"].ASHP_supply_temp
            for d in range(param["n_clusters"]):
                for t in range(24):
                    COP[d][t] = eta_carnot * (supply_temp+273.15)/(supply_temp-param["T_air"][d][t])
            devs["HP"]["COP"] = COP
            
        elif all_models["HeatPump"].COP_is_time_series:
            try:
                COP_unclustered = np.loadtxt(all_models["HeatPump"].file_time_series)
                flags["HeatPump_invalid_file"] = (len(COP_unclustered) == 8760)
                
                # Cluster COP time series
                COP_clustered = np.zeros((param["n_clusters"], 24))
                for d in range(param["n_clusters"]):
                    for t in range(24):
                        COP_clustered[d][t] = COP_unclustered[24*typedays[d]+t]
                # Replace original time series with the clustered one
                devs["HP"]["COP"] = COP_clustered
            except:
                flags["HeatPump_invalid_file"] = False
    else:
        devs["HP"]["COP"] = np.ones((param["n_clusters"], 24))
    
    if not all(val == True for val in flags.values()):
        return {}, {}, {}, {}, flags

    # Electric boiler
    devs["EB"] = {
        "feasible":  all_models["ElectricBoiler"].enabled,
        "inv_var":   all_models["ElectricBoiler"].inv_var,
        "eta_th":    all_models["ElectricBoiler"].eta_th / 100,
        "life_time": all_models["ElectricBoiler"].life_time,
        "cost_om":   all_models["ElectricBoiler"].cost_om / 100,
        "min_cap":   all_models["ElectricBoiler"].min_cap,
        "max_cap":   all_models["ElectricBoiler"].max_cap,
    }
        
    # Compression chiller
    devs["CC"] = {
        "feasible":  all_models["CompressionChiller"].enabled,
        "inv_var":   all_models["CompressionChiller"].inv_var,
        "COP":       all_models["CompressionChiller"].COP,
        "life_time": all_models["CompressionChiller"].life_time,
        "cost_om":   all_models["CompressionChiller"].cost_om / 100,
        "min_cap":   all_models["CompressionChiller"].min_cap,
        "max_cap":   all_models["CompressionChiller"].max_cap,
    }

    # Absorption chiller
    devs["AC"] = {
        "feasible":  all_models["AbsorptionChiller"].enabled,
        "inv_var":   all_models["AbsorptionChiller"].inv_var,
        "eta_th":    all_models["AbsorptionChiller"].eta_th,
        "life_time": all_models["AbsorptionChiller"].life_time,
        "cost_om":   all_models["AbsorptionChiller"].cost_om / 100,
        "min_cap":   all_models["AbsorptionChiller"].min_cap,
        "max_cap":   all_models["AbsorptionChiller"].max_cap,
    }
    
    
    ### Biomass and waste ###
    
    # Biomass CHP
    devs["BCHP"] = {
        "feasible":  all_models["BiomassCHP"].enabled,
        "inv_var":   all_models["BiomassCHP"].inv_var,
        "eta_el":    all_models["BiomassCHP"].eta_el / 100,
        "eta_th":    all_models["BiomassCHP"].eta_th / 100,
        "life_time": all_models["BiomassCHP"].life_time,
        "cost_om":   all_models["BiomassCHP"].cost_om / 100,
        "min_cap":   all_models["BiomassCHP"].min_cap,
        "max_cap":   all_models["BiomassCHP"].max_cap,
    }
    
    # Biomass boiler
    devs["BBOI"] = {
        "feasible":  all_models["BiomassBoiler"].enabled,
        "inv_var":   all_models["BiomassBoiler"].inv_var,
        "eta_th":    all_models["BiomassBoiler"].eta_th / 100,
        "life_time": all_models["BiomassBoiler"].life_time,
        "cost_om":   all_models["BiomassBoiler"].cost_om / 100,
        "min_cap":   all_models["BiomassBoiler"].min_cap,
        "max_cap":   all_models["BiomassBoiler"].max_cap,
    }
    
    # Waste CHP
    devs["WCHP"] = {
        "feasible":  all_models["WasteCHP"].enabled,
        "inv_var":   all_models["WasteCHP"].inv_var,
        "eta_el":    all_models["WasteCHP"].eta_el / 100,
        "eta_th":    all_models["WasteCHP"].eta_th / 100,
        "life_time": all_models["WasteCHP"].life_time,
        "cost_om":   all_models["WasteCHP"].cost_om / 100,
        "min_cap":   all_models["WasteCHP"].min_cap,
        "max_cap":   all_models["WasteCHP"].max_cap,
    }
    
    # Waste boiler
    devs["WBOI"] = {
        "feasible":  all_models["WasteBoiler"].enabled,
        "inv_var":   all_models["WasteBoiler"].inv_var,
        "eta_th":    all_models["WasteBoiler"].eta_th / 100,
        "life_time": all_models["WasteBoiler"].life_time,
        "cost_om":   all_models["WasteBoiler"].cost_om / 100,
        "min_cap":   all_models["WasteBoiler"].min_cap,
        "max_cap":   all_models["WasteBoiler"].max_cap,
    }
    
    ### Hydrogen ###
    
    # Electrolyzer
    devs["ELYZ"] = {
        "feasible":  all_models["Electrolyzer"].enabled,
        "inv_var":   all_models["Electrolyzer"].inv_var,
        "eta_el":    all_models["Electrolyzer"].eta_el / 100,
        "life_time": all_models["Electrolyzer"].life_time,
        "cost_om":   all_models["Electrolyzer"].cost_om / 100,
        "min_cap":   all_models["Electrolyzer"].min_cap,
        "max_cap":   all_models["Electrolyzer"].max_cap,
    }
    
    # Fuel cell
    devs["FC"] = {
        "feasible":  all_models["FuelCell"].enabled,
        "inv_var":   all_models["FuelCell"].inv_var,
        "eta_el":    all_models["FuelCell"].eta_el / 100,
        "eta_th":    all_models["FuelCell"].eta_th / 100,
        "life_time": all_models["FuelCell"].life_time,
        "cost_om":   all_models["FuelCell"].cost_om / 100,
        "min_cap":   all_models["FuelCell"].min_cap,
        "max_cap":   all_models["FuelCell"].max_cap,
        "enable_heat_diss": all_models["FuelCell"].enable_heat_diss,
    }
    
    # Hydrogen storage
    devs["H2S"] = {
        "feasible":  all_models["H2Storage"].enabled,
        "inv_var":   all_models["H2Storage"].inv_var,
        "sto_loss":  0,
        "life_time": all_models["H2Storage"].life_time,
        "cost_om":   all_models["H2Storage"].cost_om / 100,
        "min_cap":   all_models["H2Storage"].min_cap,
        "max_cap":   all_models["H2Storage"].max_cap,
    }
    
    # Sabatier reactor
    devs["SAB"] = {
        "feasible":  all_models["SabatierReactor"].enabled,
        "inv_var":   all_models["SabatierReactor"].inv_var,
        "eta":       all_models["SabatierReactor"].eta / 100,
        "life_time": all_models["SabatierReactor"].life_time,
        "cost_om":   all_models["SabatierReactor"].cost_om / 100,
        "min_cap":   all_models["SabatierReactor"].min_cap,
        "max_cap":   all_models["SabatierReactor"].max_cap,
    }
    
    
    ### Storages ###
    
    # Heat thermal energy storage
    devs["TES"] = {
        "feasible":  all_models["HeatStorage"].enabled,
        "inv_var":   all_models["HeatStorage"].inv_var / (param["rho_w"] * param["c_w"] * all_models["HeatStorage"].delta_T / 3600), # EUR/kWh
        "sto_loss":  all_models["HeatStorage"].sto_loss / 100,
        "life_time": all_models["HeatStorage"].life_time,
        "cost_om":   all_models["HeatStorage"].cost_om / 100,
        "min_cap":   all_models["HeatStorage"].min_vol * param["rho_w"] * param["c_w"] * all_models["HeatStorage"].delta_T / 3600, # kWh
        "max_cap":   all_models["HeatStorage"].max_vol * param["rho_w"] * param["c_w"] * all_models["HeatStorage"].delta_T / 3600, # kWh 
        "delta_T":   all_models["HeatStorage"].delta_T, # K
        "soc_init":  0.5,  # ---,              maximum initial state of charge
    }

    # Cold thermal energy storage
    devs["CTES"] = {
        "feasible":  all_models["ColdStorage"].enabled,
        "inv_var":   all_models["ColdStorage"].inv_var / (param["rho_w"] * param["c_w"] * all_models["ColdStorage"].delta_T / 3600), # EUR/kWh
        "sto_loss":  all_models["ColdStorage"].sto_loss / 100,
        "life_time": all_models["ColdStorage"].life_time,
        "cost_om":   all_models["ColdStorage"].cost_om / 100,
        "min_cap":   all_models["ColdStorage"].min_vol * param["rho_w"] * param["c_w"] * all_models["ColdStorage"].delta_T / 3600, # kWh
        "max_cap":   all_models["ColdStorage"].max_vol * param["rho_w"] * param["c_w"] * all_models["ColdStorage"].delta_T / 3600, # kWh
        "delta_T":   all_models["ColdStorage"].delta_T, # K,
        "soc_init":  0.5,  # ---,              maximum initial state of charge
    }

    # Battery
    devs["BAT"] = {
        "feasible":  all_models["Battery"].enabled,
        "inv_var":   all_models["Battery"].inv_var,
        "life_time": all_models["Battery"].life_time,
        "cost_om":   all_models["Battery"].cost_om / 100,
        "min_cap":   all_models["Battery"].min_cap,
        "max_cap":   all_models["Battery"].max_cap,
        "sto_loss":  0,    # 1/h,              standby losses over one time step
        "soc_init":  0.5,  # ---,              maximum initial state of charge
    }

    # Gas storage
    devs["GS"] = {
        "feasible":  all_models["GasStorage"].enabled,
        "inv_var":   all_models["GasStorage"].inv_var, # EUR/kWh
        "life_time": all_models["GasStorage"].life_time,
        "cost_om":   all_models["GasStorage"].cost_om / 100,
        "min_cap":   all_models["GasStorage"].min_cap, # kWh
        "max_cap":   all_models["GasStorage"].max_cap, # kWh 
        "sto_loss":  0,    # 1/h,              standby losses over one time step
        "soc_init":  0.5,  # ---,              maximum initial state of charge
    }
    
    
    ################################################################
    # LOAD MODEL PARAMETERS

    ### Energy costs ###
    # Electricity costs
    param["enable_supply_el"] = all_models["EnergyCosts"].enable_supply_el
    param["price_supply_el"]  = all_models["EnergyCosts"].price_supply_el
    
    if all_models["EnergyCosts"].enable_price_cap_el:
        param["price_cap_el"] =  all_models["EnergyCosts"].price_cap_el  # kEUR/MW
        result_dict["enable_price_cap_el"] = True  # for if-statement in result.html
    else:
        param["price_cap_el"] = 0   
        result_dict["enable_price_cap_el"] = False    
    param["enable_feed_in_el"]  = all_models["EnergyCosts"].enable_feed_in_el
    param["revenue_feed_in_el"] = all_models["EnergyCosts"].revenue_feed_in_el
    
    param["enable_cap_limit_el"] = all_models["EnergyCosts"].enable_cap_limit_el
    param["cap_limit_el"]        = all_models["EnergyCosts"].cap_limit_el  # kW   
        
    param["enable_supply_limit_el"] = all_models["EnergyCosts"].enable_supply_limit_el
    param["supply_limit_el"]        = all_models["EnergyCosts"].supply_limit_el * 1000 # kWh/a
            
        
    # Natural gas
    param["enable_supply_gas"] = all_models["EnergyCosts"].enable_supply_gas
    param["price_supply_gas"]  = all_models["EnergyCosts"].price_supply_gas
    if all_models["EnergyCosts"].enable_price_cap_gas:
        param["price_cap_gas"] = all_models["EnergyCosts"].price_cap_gas
        result_dict["enable_price_cap_gas"] = True  # for if-statement in result.html
    else:
        param["price_cap_gas"] = 0
        result_dict["enable_price_cap_gas"] = False        
    param["enable_feed_in_gas"]      = all_models["EnergyCosts"].enable_feed_in_gas
    param["revenue_feed_in_gas"]     = all_models["EnergyCosts"].revenue_feed_in_gas
    
    param["enable_cap_limit_gas"]    = all_models["EnergyCosts"].enable_cap_limit_gas
    param["cap_limit_gas"]           = all_models["EnergyCosts"].cap_limit_gas  # kW    
    param["enable_supply_limit_gas"] = all_models["EnergyCosts"].enable_supply_limit_gas
    param["supply_limit_gas"]        = all_models["EnergyCosts"].supply_limit_gas * 1000 # kWh/a

    
    # Biomass
    param["enable_supply_biomass"]    = all_models["EnergyCosts"].enable_supply_biomass
    param["price_biomass"]            = all_models["EnergyCosts"].price_biomass  # EUR/kW
    param["enable_supply_limit_biom"] = all_models["EnergyCosts"].enable_supply_limit_biomass  
    param["supply_limit_biomass"]     = all_models["EnergyCosts"].supply_limit_biomass * 1000  #kWh  
    
    
    # Hydrogen
    param["enable_supply_hydrogen"]       = all_models["EnergyCosts"].enable_supply_hydrogen
    param["price_hydrogen"]               = all_models["EnergyCosts"].price_hydrogen  # EUR/kW
    param["enable_supply_limit_hydrogen"] = all_models["EnergyCosts"].enable_supply_limit_hydrogen  
    param["supply_limit_hydrogen"]        = all_models["EnergyCosts"].supply_limit_hydrogen * 1000  #kWh 
    
    
    # Waste
    param["enable_supply_waste"]       = all_models["EnergyCosts"].enable_supply_waste
    param["price_waste"]               = all_models["EnergyCosts"].price_waste  # EUR/kW
    param["enable_supply_limit_waste"] = all_models["EnergyCosts"].enable_supply_limit_waste  
    param["supply_limit_waste"]        = all_models["EnergyCosts"].supply_limit_waste * 1000  #kWh  
    
    
    ### Ecological impact ###
    param["co2_tax"]         = all_models["EcologicalImpact"].co2_tax         / 1000  # EUR/kg  
    param["co2_el_grid"]     = all_models["EcologicalImpact"].co2_el_grid     / 1000  # kg/kWh
    param["co2_el_feed_in"]  = all_models["EcologicalImpact"].co2_el_feed_in  / 1000  # kg/kWh
    param["co2_gas"]         = all_models["EcologicalImpact"].co2_gas         / 1000  # kg/kWh
    param["co2_gas_feed_in"] = all_models["EcologicalImpact"].co2_gas_feed_in / 1000  # kg/kWh
    param["co2_biom"]        = all_models["EcologicalImpact"].co2_biomass     / 1000  # kg/kWh
    param["co2_waste"]       = all_models["EcologicalImpact"].co2_waste       / 1000  # kg/kWh
    param["co2_hydrogen"]    = all_models["EcologicalImpact"].co2_hydrogen    / 1000  # kg/kWh

    
    ### General ###
    param["optim_focus"]       = all_models["General"].optim_focus 
    param["interest_rate"]     = all_models["General"].interest_rate / 100
    param["observation_time"]  = int(all_models["General"].project_lifetime)
    param["peak_dem_met_conv"] = all_models["General"].peak_dem_met_conv 
    # Number of design days assigned on the top of this script: 'all_models["General"].design_days'
    
    ### Reference scenario ###
    param["ref"] = {}
    param["ref"]["enable_hp"]  = all_models["ReferenceCase"].enable_ref_hp
    param["ref"]["cop_hp"]     = all_models["ReferenceCase"].cop_ref_hp
    param["ref"]["cop_cc"]     = all_models["ReferenceCase"].cop_ref_cc
    param["ref"]["enable_chp"] = all_models["ReferenceCase"].enable_ref_chp
    
    
    ### Catch common input errors of user ###
    
    # Electricity feed-in revenue larger than electricity price
    if param["enable_feed_in_el"] and not param["enable_supply_limit_el"]: 
        if param["revenue_feed_in_el"] > param["price_supply_el"]:
            flags["unconstrained_el_feed_in"] = False
    
    # Gas feed-in revenue larger than gas price
    if param["enable_feed_in_gas"] and not param["enable_supply_limit_gas"]: 
        if param["revenue_feed_in_gas"] > param["price_supply_gas"]:
            flags["unconstrained_gas_feed_in"] = False
    
    
    ################################################################
    # INITIALIZE CALCULATION

    # Calculate annual investments
    devs, param = calc_annual_investment(devs, param)
    
    # Calculate reference scenario
    result_dict = calc_reference(devs, dem, param, dem_uncl, result_dict)

    # print("=== PARAMETER ===")
    # print(param)
    # print("=== TECHNOLOGIES ===")
    # print(devs)
    
    # Calculate values for post-processing
    result_dict = calc_monthly_dem(dem_uncl, param_uncl, result_dict)

    return param, devs, dem, result_dict, flags

    

#%% SUB-FUNCTIONS ##################################################


def calc_reference(devs, dem, param, dem_uncl, result_dict):
    """
    Calculation of reference scenario/system.
    """
    
    days = range(param["n_clusters"])
    time_steps = range(24)
    
    el_chp = {}
    gas_chp = {}
    heat_chp= {}
    el_hp = {}
    heat_hp = {}
    heat_boiler = {}
    gas_boiler = {}
    cool_chiller = {}
    el_chiller = {}
    h2_elyz = {}
    el_elyz = {}
    el_dem = {}
    gas_dem = {}
    for d in days:
        el_chp[d] = np.zeros(24)
        gas_chp[d] = np.zeros(24)
        heat_chp[d]= np.zeros(24)
        el_hp[d] = np.zeros(24)
        heat_hp[d] = np.zeros(24)
        heat_boiler[d] = np.zeros(24)
        gas_boiler[d] = np.zeros(24)
        cool_chiller[d] = np.zeros(24)
        h2_elyz[d] = np.zeros(24)
        el_elyz[d] = np.zeros(24)
        el_chiller[d] = np.zeros(24)
        el_dem[d] = np.zeros(24)
        gas_dem[d] = np.zeros(24)
        for t in time_steps:
        
            # Chiller
            cool_chiller[d][t] = dem["cool"][d][t]
            el_chiller[d][t] = cool_chiller[d][t] / param["ref"]["cop_cc"]
            
            # Electrolyzer
            h2_elyz[d][t] = dem["hydrogen"][d][t]
            el_elyz[d][t] = h2_elyz[d][t] / devs["ELYZ"]["eta_el"]
            
            # Total electricity demand to be covered in each time step
            total_power_dem = dem["power"][d][t] + el_chiller[d][t] + el_elyz[d][t]
            
            # Case 1: CHP and HP enabled
            if param["ref"]["enable_chp"] and param["ref"]["enable_hp"]: 
                #print("Case 1: CHP and HP enabled")
                    
                # Boiler not operated
                heat_boiler[d][t] = 0
                gas_boiler[d][t] = 0
                    
                hp_runs = (dem["heat"][d][t] > (total_power_dem/devs["CHP"]["eta_el"]*devs["CHP"]["eta_th"]))
                    
                if hp_runs:
                    #print("Heat pump runs")
                    el_hp[d][t] = (dem["heat"][d][t] - (total_power_dem/devs["CHP"]["eta_el"]*devs["CHP"]["eta_th"])) / ((devs["CHP"]["eta_th"]/devs["CHP"]["eta_el"]) + param["ref"]["cop_hp"])
                    heat_hp[d][t] = param["ref"]["cop_hp"] * el_hp[d][t]
                    el_chp[d][t] = el_hp[d][t] + total_power_dem
                    gas_chp[d][t] = el_chp[d][t] / devs["CHP"]["eta_el"]
                    heat_chp[d][t] = gas_chp[d][t] * devs["CHP"]["eta_th"]
                    el_dem[d][t] = 0
                    
                else:  # if HP is not running
                    #print("Heat pump does not run")
                    # Heat pump not operated
                    el_hp[d][t] = 0
                    heat_hp[d][t] = 0
                    
                    # CHP runs at part-load and covers heating demand exactly, electricity drawn from grid
                    heat_chp[d][t] = dem["heat"][d][t]
                    gas_chp[d][t] = heat_chp[d][t] / devs["CHP"]["eta_th"]
                    el_chp[d][t] = gas_chp[d][t] * devs["CHP"]["eta_el"]
                    
            # Case 2: CHP and BOI enabled
            elif param["ref"]["enable_chp"] and (not param["ref"]["enable_hp"]):
                #print("Case 2: CHP and BOI enabled")
                # Heat pump not operated
                el_hp[d][t] = 0
                heat_hp[d][t] = 0
                
                # check if CHP covers heat demand alone
                boi_runs = (total_power_dem / devs["CHP"]["eta_el"] * devs["CHP"]["eta_th"]) < dem["heat"][d][t]
                
                # CHP runs at part-load due to low heating demand, electricity drawn from grid
                if boi_runs:
                    #print("Boiler runs")
                    el_chp[d][t] = total_power_dem
                    gas_chp[d][t] = total_power_dem / devs["CHP"]["eta_el"]
                    heat_chp[d][t] = gas_chp[d][t] * devs["CHP"]["eta_th"]
                    heat_boiler[d][t] = dem["heat"][d][t] - heat_chp[d][t]
                    gas_boiler[d][t] = heat_boiler[d][t] / devs["BOI"]["eta_th"]
                    
                else:  # if BOI is not running
                    #print("Boiler does not run")
                    heat_boiler[d][t] = 0
                    gas_boiler[d][t] = 0
                    
                    heat_chp[d][t] = dem["heat"][d][t]
                    gas_chp[d][t] = heat_chp[d][t] / devs["CHP"]["eta_th"]
                    el_chp[d][t] = gas_chp[d][t] * devs["CHP"]["eta_el"]
                    
                
            # Case 3: HP enabled (no CHP)
            elif (not param["ref"]["enable_chp"]) and param["ref"]["enable_hp"]:
                #print("Case 3: HP enabled (no CHP)")
                # CHP and boiler not operated            
                el_chp[d][t] = 0
                gas_chp[d][t] = 0
                heat_chp[d][t] = 0
                heat_boiler[d][t] = 0
                gas_boiler[d][t] = 0
            
                # Heat pump
                heat_hp[d][t] = dem["heat"][d][t]
                el_hp[d][t] = heat_hp[d][t] / param["ref"]["cop_hp"]
            
            # Case 4: BOI enabled (no CHP)
            elif (not param["ref"]["enable_chp"]) and (not param["ref"]["enable_hp"]):
                #print("Case 4: BOI enabled (no CHP)")
                # CHP and heat pump not operated            
                el_chp[d][t] = 0
                gas_chp[d][t] = 0
                heat_chp[d][t] = 0
                el_hp[d][t] = 0
                heat_hp[d][t] = 0
            
                # Gas boiler
                heat_boiler[d][t] = dem["heat"][d][t]                
                gas_boiler[d][t] = heat_boiler[d][t] / devs["BOI"]["eta_th"]
                
            
            # Sum up
            el_dem[d][t] = dem["power"][d][t] + el_chiller[d][t] + el_hp[d][t] + el_elyz[d][t] - el_chp[d][t]
            gas_dem[d][t] = gas_boiler[d][t] + gas_chp[d][t]
            
            
            #print("dem[heat][d][t] = " + str(dem["heat"][d][t]))
            #print("total_power_dem = " + str(total_power_dem))
            #print("el_chp[d][t] = " + str(el_chp[d][t]))
            #print("heat_chp[d][t] = " + str(heat_chp[d][t]))
            
            #print("el_hp[d][t] = " + str(el_hp[d][t]))
            #print("heat_hp[d][t] = " + str(heat_hp[d][t]))
            
            #print("heat_boiler[d][t] = " + str(heat_boiler[d][t]))
            
            #print("gas_boiler[d][t] = " + str(gas_boiler[d][t]))
            #print("gas_chp[d][t] = " + str(gas_chp[d][t]))
            
            #print("el_dem[d][t] = " + str(el_dem[d][t]))
            #print("gas_dem[d][t] = " + str(gas_dem[d][t]))
    
    # Sum up
    result_dict["ref"] = {}
    result_dict["ref"]["heat_hp_total"] = int(sum(sum(heat_hp[d][t] for t in time_steps) * param["day_weights"][d] for d in days)/1000)
    result_dict["ref"]["el_chp_total"] = int(sum(sum(el_chp[d][t] for t in time_steps) * param["day_weights"][d] for d in days)/1000)    
    result_dict["ref"]["heat_boi_total"] = int(sum(sum(heat_boiler[d][t] for t in time_steps) * param["day_weights"][d] for d in days)/1000)    
    result_dict["ref"]["cool_cc_total"] = int(sum(sum(cool_chiller[d][t] for t in time_steps) * param["day_weights"][d] for d in days)/1000) 
    result_dict["ref"]["hydrogen_elyz_total"] = int(sum(sum(h2_elyz[d][t] for t in time_steps) * param["day_weights"][d] for d in days)/1000) 
    result_dict["ref"]["el_dem_total"] = int(sum(sum(el_dem[d][t] for t in time_steps) * param["day_weights"][d] for d in days)/1000)
    result_dict["ref"]["gas_dem_total"] = int(sum(sum(gas_dem[d][t] for t in time_steps) * param["day_weights"][d] for d in days)/1000)

    
    ### DETERMINE CAPACITIES ###
    # Calculate all flows for the unclustered time series again, and then determine the capacities
    
    el_chp_uncl = np.zeros(8760)
    gas_chp_uncl = np.zeros(8760)
    heat_chp_uncl = np.zeros(8760)
    el_hp_uncl = np.zeros(8760)
    heat_hp_uncl = np.zeros(8760)
    heat_boiler_uncl = np.zeros(8760)
    gas_boiler_uncl = np.zeros(8760)
    el_elyz_uncl = np.zeros(8760)
    el_chiller_uncl = np.zeros(8760)
    el_dem_uncl = np.zeros(8760)
    gas_dem_uncl = np.zeros(8760)
    
    for t in range(8760):
        
        # Chiller and electrolyzer
        el_chiller_uncl[t] = dem_uncl["cool"][t] / param["ref"]["cop_cc"]
        el_elyz_uncl[t] = dem_uncl["hydrogen"][t] / devs["ELYZ"]["eta_el"]
        
        # Total electricity demand to be covered in each time step
        total_power_dem = dem_uncl["power"][t] + el_chiller_uncl[t] + el_elyz_uncl[t]
        
        # Case 1: CHP and HP enabled
        if param["ref"]["enable_chp"] and param["ref"]["enable_hp"]: 
            #print("Case 1: CHP and HP enabled")
                
            # Boiler not operated
            heat_boiler_uncl[t] = 0
            gas_boiler_uncl[t] = 0
                
            hp_runs = (dem_uncl["heat"][t] > (total_power_dem/devs["CHP"]["eta_el"]*devs["CHP"]["eta_th"]))
    
            if hp_runs:
                #print("Heat pump runs")
                el_hp_uncl[t] = (dem_uncl["heat"][t] - (total_power_dem/devs["CHP"]["eta_el"]*devs["CHP"]["eta_th"])) / ((devs["CHP"]["eta_th"]/devs["CHP"]["eta_el"]) + param["ref"]["cop_hp"])
                heat_hp_uncl[t] = param["ref"]["cop_hp"] * el_hp_uncl[t]
                el_chp_uncl[t] = el_hp_uncl[t] + total_power_dem
                gas_chp_uncl[t] = el_chp_uncl[t] / devs["CHP"]["eta_el"]
                heat_chp_uncl[t] = gas_chp_uncl[t] * devs["CHP"]["eta_th"]
                el_dem_uncl[t] = 0
                
            else:  # if HP is not running
                #print("Heat pump does not run")
                # Heat pump not operated
                el_hp_uncl[t] = 0
                heat_hp_uncl[t] = 0
                
                # CHP runs at part-load and covers heating demand exactly, electricity drawn from grid
                heat_chp_uncl[t] = dem_uncl["heat"][t]
                gas_chp_uncl[t] = heat_chp_uncl[t] / devs["CHP"]["eta_th"]
                el_chp_uncl[t] = gas_chp_uncl[t] * devs["CHP"]["eta_el"]
                
        # Case 2: CHP and BOI enabled
        elif param["ref"]["enable_chp"] and (not param["ref"]["enable_hp"]):
            #print("Case 2: CHP and BOI enabled")
            # Heat pump not operated
            el_hp_uncl[t] = 0
            heat_hp_uncl[t] = 0
            
            # check if CHP covers heat demand alone
            boi_runs = (total_power_dem / devs["CHP"]["eta_el"] * devs["CHP"]["eta_th"]) < dem_uncl["heat"][t]
            
            # CHP runs at part-load due to low heating demand, electricity drawn from grid
            if boi_runs:
                #print("Boiler runs")
                el_chp_uncl[t] = total_power_dem
                gas_chp_uncl[t] = total_power_dem / devs["CHP"]["eta_el"]
                heat_chp_uncl[t] = gas_chp_uncl[t] * devs["CHP"]["eta_th"]
                heat_boiler_uncl[t] = dem_uncl["heat"][t] - heat_chp_uncl[t]
                gas_boiler_uncl[t] = heat_boiler_uncl[t] / devs["BOI"]["eta_th"]
                
            else:  # if BOI is not running
                #print("Boiler does not run")
                heat_boiler_uncl[t] = 0
                gas_boiler_uncl[t] = 0
                
                heat_chp_uncl[t] = dem_uncl["heat"][t]
                gas_chp_uncl[t] = heat_chp_uncl[t] / devs["CHP"]["eta_th"]
                el_chp_uncl[t] = gas_chp_uncl[t] * devs["CHP"]["eta_el"]
                
            
        # Case 3: HP enabled (no CHP)
        elif (not param["ref"]["enable_chp"]) and param["ref"]["enable_hp"]:
            #print("Case 3: HP enabled (no CHP)")
            # CHP and boiler not operated            
            el_chp_uncl[t] = 0
            gas_chp_uncl[t] = 0
            heat_chp_uncl[t] = 0
            heat_boiler_uncl[t] = 0
            gas_boiler_uncl[t] = 0
        
            # heat pump
            heat_hp_uncl[t] = dem_uncl["heat"][t]
            el_hp_uncl[t] = heat_hp_uncl[t] / param["ref"]["cop_hp"]
        
        # Case 4: BOI enabled (no CHP)
        elif (not param["ref"]["enable_chp"]) and (not param["ref"]["enable_hp"]):
            #print("Case 4: BOI enabled (no CHP)")
            # CHP and heat pump not operated            
            el_chp_uncl[t] = 0
            gas_chp_uncl[t] = 0
            heat_chp_uncl[t] = 0
            el_hp_uncl[t] = 0
            heat_hp_uncl[t] = 0
        
            # Gas boiler
            heat_boiler_uncl[t] = dem_uncl["heat"][t]                
            gas_boiler_uncl[t] = heat_boiler_uncl[t] / devs["BOI"]["eta_th"]
            
        
        # Sum up
        el_dem_uncl[t] = dem_uncl["power"][t] + el_chiller_uncl[t] + el_hp_uncl[t] + el_elyz_uncl[t] - el_chp_uncl[t]
        gas_dem_uncl[t] = gas_boiler_uncl[t] + gas_chp_uncl[t]
    
    # Capacity is maximum of generation (of unclustered time series)
    result_dict["ref"]["chp_cap"] = int(np.max(el_chp_uncl))
    result_dict["ref"]["hp_cap"] = int(np.max(heat_hp_uncl))
    result_dict["ref"]["boiler_cap"] = int(np.max(heat_boiler_uncl))
    result_dict["ref"]["chiller_cap"] = int(np.max(dem_uncl["cool"]))
    result_dict["ref"]["elyz_cap"] = int(np.max(dem_uncl["hydrogen"]) / devs["ELYZ"]["eta_el"])
    
    # Electricity
    result_dict["ref"]["el_grid_cap"] = np.max([np.max(el_dem[d]) for d in days])
    result_dict["ref"]["gas_grid_cap"] = np.max([np.max(gas_dem[d]) for d in days])
    
    result_dict["ref"]["gas_costs"] = result_dict["ref"]["gas_grid_cap"] * param["price_cap_gas"] + result_dict["ref"]["gas_dem_total"]* 1000 * param["price_supply_gas"] # cap price and supply price
    
    result_dict["ref"]["el_costs"] = result_dict["ref"]["el_grid_cap"] * param["price_cap_el"] + result_dict["ref"]["el_dem_total"]*1000 * param["price_supply_el"] # cap price and supply price    
    
    # CO2 tax
    result_dict["ref"]["co2_costs"] = result_dict["ref"]["gas_dem_total"] * param["co2_gas"] * param["co2_tax"] * 1000
    
    result_dict["ref"]["invest_om"] = (
        (devs["CHP"]["ann_factor"] + devs["CHP"]["cost_om"]) * devs["CHP"]["inv_var"] * result_dict["ref"]["chp_cap"] 
        + (devs["BOI"]["ann_factor"] + devs["BOI"]["cost_om"]) * devs["BOI"]["inv_var"] * result_dict["ref"]["boiler_cap"]
        + (devs["HP"]["ann_factor"] + devs["HP"]["cost_om"]) * devs["HP"]["inv_var"] * result_dict["ref"]["hp_cap"]
        + (devs["CC"]["ann_factor"] + devs["CC"]["cost_om"]) * devs["CC"]["inv_var"] * result_dict["ref"]["chiller_cap"]
        + (devs["ELYZ"]["ann_factor"] + devs["ELYZ"]["cost_om"]) * devs["ELYZ"]["inv_var"] * result_dict["ref"]["elyz_cap"])
        
    result_dict["ref"]["tac"] = int(result_dict["ref"]["invest_om"] 
                               + result_dict["ref"]["gas_costs"] 
                               + result_dict["ref"]["el_costs"]
                               + result_dict["ref"]["co2_costs"])
                              
    result_dict["ref"]["co2"] = int(result_dict["ref"]["gas_dem_total"] * param["co2_gas"] + result_dict["ref"]["el_dem_total"] * param["co2_el_grid"])  # t/a  (= t/MWh x MWh)
    
    # For result page listing of technologies
    result_dict["ref"]["enable_chp"] = param["ref"]["enable_chp"]
    result_dict["ref"]["enable_hp"] = param["ref"]["enable_hp"]
    
    return result_dict


    
def calc_annual_investment(devs, param):
    """
    Calculation of total investment costs including replacements (based on VDI 2067-1, pages 16-17).

    Parameters
    ----------
    dev : dictionary
        technology parameter
    param : dictionary
        economic parameters

    Returns
    -------
    annualized fix and variable investment
    """

    observation_time = param["observation_time"]
    interest_rate = param["interest_rate"]
    q = 1 + param["interest_rate"]

    # Calculate capital recovery factor
    CRF = ((q**observation_time)*interest_rate)/((q**observation_time)-1)
    
    # Calculate annuity factor for each device
    for device in devs.keys():
        
        # Get device life time
        life_time = devs[device]["life_time"]

        # Number of required replacements
        n = int(math.floor(observation_time / life_time))
        
        # Investment for replacements
        invest_replacements = sum((q ** (-i * life_time)) for i in range(1, n+1))

        # Residual value of final replacement
        res_value = ((n+1) * life_time - observation_time) / life_time * (q ** (-observation_time))

        # Calculate annualized investments       
        if life_time > observation_time:
            devs[device]["ann_factor"] = (1 - res_value) * CRF 
        else:
            devs[device]["ann_factor"] = ( 1 + invest_replacements - res_value) * CRF 
            
    # Save capital recovery factor
    param["CRF"] = CRF

    return devs, param

    
    
def calc_monthly_dem(dem_uncl, param_uncl, result_dict):

    month_tuple = ("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
    days_sum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    monthly_dem = {}
    year_peak = {}
    year_sum = {}
    for m in ["heat", "cool", "power", "hydrogen"]:
        monthly_dem[m] = {}
        year_peak[m] = int(np.max(dem_uncl[m]))
        year_sum[m] = int(np.sum(dem_uncl[m]) / 1000)
        for month in range(12):
            monthly_dem[m][month_tuple[month]] = sum(dem_uncl[m][t] for t in range(days_sum[month]*24, days_sum[month+1]*24)) / 1000
    
    result_dict["monthly_dem"] = monthly_dem
    result_dict["year_peak"] = year_peak
    result_dict["year_sum"] = year_sum
    
    
    #monthly_val = {}
    #year_peak = {}
    #year_sum = {}
    #for m in ["T_air", "GHI"]:  # "wind_speed"]:
    #    monthly_val[m] = {}
    #    year_peak[m] = int(np.max(param_uncl[m]))
    #    year_sum[m] = int(np.sum(param_uncl[m]) / 1000)
    #    for month in range(12):
    #        monthly_val[m][month_tuple[month]] = sum(param_uncl[m][t] for t in range(days_sum[month]*24, #days_sum[month+1]*24)) / 1000
            
    #result_dict["monthly_val"] = monthly_val
    
    return result_dict

    
def calc_WT_power(devs, param):
    """
    According to data sheet of wind turbine Enercon E40.
    """

    power_curve = {0:  (0.0,    0.00),
                   1:  (2.4,    0.00),
                   2:  (2.5,    1.14),
                   3:  (3.0,    4.37),
                   4:  (3.5,   10.64),
                   5:  (4.0,   18.87),
                   6:  (4.5,   29.77),
                   7:  (5.0,   40.39),
                   8:  (5.5,   52.85),
                   9:  (6.0,   69.36),
                   10: (6.5,   88.02),
                   11: (7,    112.19),
                   12: (7.5,  134.67),
                   13: (8,    165.38),
                   14: (8.5,  197.08),
                   15: (9,    236.89),
                   16: (9.5,  279.46),
                   17: (10,   328.00),
                   18: (10.5, 362.93),
                   19: (11,   396.64),
                   20: (11.5, 435.27),
                   21: (12,   465.15),
                   22: (12.5, 483.63),
                   23: (13,   495.95),
                   24: (14,   500.00),
                   25: (25,   500.00),
                   26: (25.1,   0.00),
                   27: (1000,   0.00),
                   }

    wind_speed_corr = param["wind_speed"]*(devs["WT"]["hub_h"]/devs["WT"]["ref_h"]) ** devs["WT"]["h_coeff"]  # kW
    
    WT_power = np.zeros(np.shape(wind_speed_corr))
    for d in range(param["n_clusters"]):
        for t in range(24):
            WT_power[d][t] = get_turbine_power(wind_speed_corr[d][t], power_curve)
    
    WT_power_norm = WT_power / 500  # power_curve with 500 kW as maximum output
    
    return WT_power_norm
    
    
def get_turbine_power(wind_speed, power_curve):
    if wind_speed <= 0:
        return 0
    if wind_speed > power_curve[len(power_curve)-1][0]:
        print("Error: Wind speed is " + str(wind_speed) + " m/s and exceeds wind power curve table.")
        return 0
    
    # Linear interpolation:
    for k in range(len(power_curve)):
        if power_curve[k][0] > wind_speed:
           power = (power_curve[k][1]-power_curve[k-1][1])/(power_curve[k][0]-power_curve[k-1][0]) * (wind_speed-power_curve[k-1][0]) + power_curve[k-1][1]
           break
    return power

    