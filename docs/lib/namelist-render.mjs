import { formatDateParts } from "./case-config.mjs";

function formatFixed(value, places = 4) {
  return Number(value).toFixed(places);
}

export function renderNamelistWps(caseConfig) {
  const fgName = caseConfig.sourceProfile.splitPrefixes ? "'SFILE', 'SOILFILE', 'FILE'" : "'FILE'";

  return `&share
 wrf_core               = 'ARW',
 max_dom                = 1,
 start_date             = '${caseConfig.startStamp}',
 end_date               = '${caseConfig.endStamp}',
 interval_seconds       = ${caseConfig.intervalSeconds},
 io_form_geogrid        = 2,
/
&geogrid
 parent_id              = 1,
 parent_grid_ratio      = 1,
 i_parent_start         = 1,
 j_parent_start         = 1,
 e_we                   = ${caseConfig.eWe},
 e_sn                   = ${caseConfig.eSn},
 geog_data_res          = 'default',
 dx                     = ${formatFixed(caseConfig.dxMeters, 1)},
 dy                     = ${formatFixed(caseConfig.dyMeters, 1)},
 map_proj               = '${caseConfig.mapProj}',
 ref_lat                = ${formatFixed(caseConfig.refLat)},
 ref_lon                = ${formatFixed(caseConfig.refLon)},
 truelat1               = ${formatFixed(caseConfig.truelat1)},
 truelat2               = ${formatFixed(caseConfig.truelat2)},
 stand_lon              = ${formatFixed(caseConfig.standLon)},
 geog_data_path         = '${caseConfig.geogDataPath}'
/
&ungrib
 out_format             = 'WPS',
 prefix                 = 'FILE',
/
&metgrid
 fg_name                = ${fgName},
 io_form_metgrid        = 2,
/
`;
}

export function renderNamelistInput(caseConfig) {
  const start = formatDateParts(caseConfig.start);
  const end = formatDateParts(caseConfig.end);
  const runDays = Math.floor(caseConfig.runHours / 24);
  const runHoursRemainder = caseConfig.runHours % 24;

  return `&time_control
 run_days                            = ${runDays},
 run_hours                           = ${runHoursRemainder},
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = ${start.year},
 start_month                         = ${start.month},
 start_day                           = ${start.day},
 start_hour                          = ${start.hour},
 end_year                            = ${end.year},
 end_month                           = ${end.month},
 end_day                             = ${end.day},
 end_hour                            = ${end.hour},
 interval_seconds                    = ${caseConfig.intervalSeconds},
 input_from_file                     = .true.,
 history_interval                    = ${caseConfig.historyMinutes},
 frames_per_outfile                  = 1,
 restart                             = .false.,
 restart_interval                    = 7200,
 io_form_history                     = 2,
 io_form_restart                     = 2,
 io_form_input                       = 2,
 io_form_boundary                    = 2,
/

&domains
 reasonable_time_step_ratio          = 10.0001011,
 time_step                           = 12,
 time_step_fract_num                 = 0,
 time_step_fract_den                 = 1,
 max_dom                             = 1,
 e_we                                = ${caseConfig.eWe},
 e_sn                                = ${caseConfig.eSn},
 e_vert                              = 45,
 dzstretch_s                         = 1.1,
 p_top_requested                     = 5000,
 num_metgrid_levels                  = ${caseConfig.numMetgridLevels},
 num_metgrid_soil_levels             = ${caseConfig.numMetgridSoilLevels},
 dx                                  = 3000,
 dy                                  = 3000,
 grid_id                             = 1,
 parent_id                           = 0,
 i_parent_start                      = 1,
 j_parent_start                      = 1,
 parent_grid_ratio                   = 1,
 parent_time_step_ratio              = 1,
 feedback                            = 1,
 smooth_option                       = 0,
/

&physics
 mp_physics                          = 10,
 cu_physics                          = 0,
 ra_lw_physics                       = 4,
 ra_sw_physics                       = 4,
 bl_pbl_physics                      = 1,
 sf_sfclay_physics                   = 1,
 sf_surface_physics                  = 1,
 radt                                = 15,
 bldt                                = 0,
 cudt                                = 0,
 isftcflx                            = 1,
 icloud                              = 1,
 do_radar_ref                        = 1,
/

&fdda
/

&dynamics
 hybrid_opt                          = 2,
 w_damping                           = 0,
 diff_opt                            = 2,
 km_opt                              = 4,
 diff_6th_opt                        = 0,
 diff_6th_factor                     = 0.12,
 base_temp                           = 290.,
 damp_opt                            = 3,
 zdamp                               = 5000.,
 dampcoef                            = 0.2,
 khdif                               = 0,
 kvdif                               = 0,
 non_hydrostatic                     = .true.,
 moist_adv_opt                       = 1,
 scalar_adv_opt                      = 1,
 gwd_opt                             = 1,
/

&bdy_control
 spec_bdy_width                      = 5,
 specified                           = .true.,
/

&grib2
/

&namelist_quilt
 nio_tasks_per_group                 = 0,
 nio_groups                          = 1,
/
`;
}
