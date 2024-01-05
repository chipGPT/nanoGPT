# Begin_DVE_Session_Save_Info
# DVE full session
# Saved on Fri Dec 22 11:49:03 2023
# Designs open: 1
#   Sim: /net/brevort/z/guanchen/projects/transformer/nanoGPT_hw/HW/SA/sim
# Toplevel windows open: 1
# 	TopLevel.3
#   Wave.2: 18 signals
#   Group count = 2
#   Group Group1 signal count = 11
#   Group Group2 signal count = 18
# End_DVE_Session_Save_Info

# DVE version: R-2020.12-SP2-1_Full64
# DVE build date: Jul 18 2021 21:21:42


#<Session mode="Full" path="/net/brevort/z/guanchen/projects/transformer/nanoGPT_hw/HW/SA/DVEfiles/session.tcl" type="Debug">

gui_set_loading_session_type Post
gui_continuetime_set

# Close design
if { [gui_sim_state -check active] } {
    gui_sim_terminate
}
gui_close_db -all
gui_expr_clear_all

# Close all windows
gui_close_window -type Console
gui_close_window -type Wave
gui_close_window -type Source
gui_close_window -type Schematic
gui_close_window -type Data
gui_close_window -type DriverLoad
gui_close_window -type List
gui_close_window -type Memory
gui_close_window -type HSPane
gui_close_window -type DLPane
gui_close_window -type Assertion
gui_close_window -type CovHier
gui_close_window -type CoverageTable
gui_close_window -type CoverageMap
gui_close_window -type CovDetail
gui_close_window -type Local
gui_close_window -type Stack
gui_close_window -type Watch
gui_close_window -type Group
gui_close_window -type Transaction



# Application preferences
gui_set_pref_value -key app_default_font -value {Helvetica,10,-1,5,50,0,0,0,0,0}
gui_src_preferences -tabstop 8 -maxbits 24 -windownumber 1
#<WindowLayout>

# DVE top-level session


# Create and position top-level window: TopLevel.3

if {![gui_exist_window -window TopLevel.3]} {
    set TopLevel.3 [ gui_create_window -type TopLevel \
       -icon $::env(DVE)/auxx/gui/images/toolbars/dvewin.xpm] 
} else { 
    set TopLevel.3 TopLevel.3
}
gui_show_window -window ${TopLevel.3} -show_state maximized -rect {{21 66} {1940 971}}

# ToolBar settings
gui_set_toolbar_attributes -toolbar {TimeOperations} -dock_state top
gui_set_toolbar_attributes -toolbar {TimeOperations} -offset 0
gui_show_toolbar -toolbar {TimeOperations}
gui_hide_toolbar -toolbar {&File}
gui_set_toolbar_attributes -toolbar {&Edit} -dock_state top
gui_set_toolbar_attributes -toolbar {&Edit} -offset 0
gui_show_toolbar -toolbar {&Edit}
gui_hide_toolbar -toolbar {CopyPaste}
gui_set_toolbar_attributes -toolbar {&Trace} -dock_state top
gui_set_toolbar_attributes -toolbar {&Trace} -offset 0
gui_show_toolbar -toolbar {&Trace}
gui_set_toolbar_attributes -toolbar {TraceInstance} -dock_state top
gui_set_toolbar_attributes -toolbar {TraceInstance} -offset 0
gui_show_toolbar -toolbar {TraceInstance}
gui_hide_toolbar -toolbar {BackTrace}
gui_set_toolbar_attributes -toolbar {&Scope} -dock_state top
gui_set_toolbar_attributes -toolbar {&Scope} -offset 0
gui_show_toolbar -toolbar {&Scope}
gui_set_toolbar_attributes -toolbar {&Window} -dock_state top
gui_set_toolbar_attributes -toolbar {&Window} -offset 0
gui_show_toolbar -toolbar {&Window}
gui_set_toolbar_attributes -toolbar {Signal} -dock_state top
gui_set_toolbar_attributes -toolbar {Signal} -offset 0
gui_show_toolbar -toolbar {Signal}
gui_set_toolbar_attributes -toolbar {Zoom} -dock_state top
gui_set_toolbar_attributes -toolbar {Zoom} -offset 0
gui_show_toolbar -toolbar {Zoom}
gui_set_toolbar_attributes -toolbar {Zoom And Pan History} -dock_state top
gui_set_toolbar_attributes -toolbar {Zoom And Pan History} -offset 0
gui_show_toolbar -toolbar {Zoom And Pan History}
gui_set_toolbar_attributes -toolbar {Grid} -dock_state top
gui_set_toolbar_attributes -toolbar {Grid} -offset 0
gui_show_toolbar -toolbar {Grid}
gui_set_toolbar_attributes -toolbar {Simulator} -dock_state top
gui_set_toolbar_attributes -toolbar {Simulator} -offset 0
gui_show_toolbar -toolbar {Simulator}
gui_set_toolbar_attributes -toolbar {Interactive Rewind} -dock_state top
gui_set_toolbar_attributes -toolbar {Interactive Rewind} -offset 0
gui_show_toolbar -toolbar {Interactive Rewind}
gui_set_toolbar_attributes -toolbar {Testbench} -dock_state top
gui_set_toolbar_attributes -toolbar {Testbench} -offset 0
gui_show_toolbar -toolbar {Testbench}

# End ToolBar settings

# Docked window settings
gui_sync_global -id ${TopLevel.3} -option true

# MDI window settings
set Wave.2 [gui_create_window -type {Wave}  -parent ${TopLevel.3}]
gui_show_window -window ${Wave.2} -show_state maximized
gui_update_layout -id ${Wave.2} {{show_state maximized} {dock_state undocked} {dock_on_new_line false} {child_wave_left 557} {child_wave_right 1357} {child_wave_colname 276} {child_wave_colvalue 276} {child_wave_col1 0} {child_wave_col2 1}}

# End MDI window settings

gui_set_env TOPLEVELS::TARGET_FRAME(Source) none
gui_set_env TOPLEVELS::TARGET_FRAME(Schematic) none
gui_set_env TOPLEVELS::TARGET_FRAME(PathSchematic) none
gui_set_env TOPLEVELS::TARGET_FRAME(Wave) none
gui_set_env TOPLEVELS::TARGET_FRAME(List) none
gui_set_env TOPLEVELS::TARGET_FRAME(Memory) none
gui_set_env TOPLEVELS::TARGET_FRAME(DriverLoad) none
gui_update_statusbar_target_frame ${TopLevel.3}

#</WindowLayout>

#<Database>

# DVE Open design session: 

if { [llength [lindex [gui_get_db -design Sim] 0]] == 0 } {
gui_set_env SIMSETUP::SIMARGS {{+v2k +vc +vcs+lic+wait +multisource_int_delays +lint=TFIPC-L +neg_tchk +overlap +warn=noSDFCOM_UHICD +warn=noSDFCOM_IWSBA +warn=noSDFCOM_IANE +warn=noSDFCOM_PONF +warn=noSDFCOM_UHICD,noSDFCOM_IWSBA,noSDFCOM_IANE,noSDFCOM_PONF +libext+.v+.vlib+.vh}}
gui_set_env SIMSETUP::SIMEXE {sim}
gui_set_env SIMSETUP::ALLOW_POLL {0}
if { ![gui_is_db_opened -db {/net/brevort/z/guanchen/projects/transformer/nanoGPT_hw/HW/SA/sim}] } {
gui_sim_run Ucli -exe sim -args { +v2k +vc +vcs+lic+wait +multisource_int_delays +lint=TFIPC-L +neg_tchk +overlap +warn=noSDFCOM_UHICD +warn=noSDFCOM_IWSBA +warn=noSDFCOM_IANE +warn=noSDFCOM_PONF +warn=noSDFCOM_UHICD,noSDFCOM_IWSBA,noSDFCOM_IANE,noSDFCOM_PONF +libext+.v+.vlib+.vh -ucligui} -dir /net/brevort/z/guanchen/projects/transformer/nanoGPT_hw/HW/SA -nosource
}
}
if { ![gui_sim_state -check active] } {error "Simulator did not start correctly" error}
gui_set_precision 1ps
gui_set_time_units 1ps
#</Database>

# DVE Global setting session: 


# Global: Breakpoints

# Global: Bus

# Global: Expressions

# Global: Signal Time Shift

# Global: Signal Compare

# Global: Signal Groups


set _session_group_1 Group1
gui_sg_create "$_session_group_1"
set Group1 "$_session_group_1"

gui_sg_addsignal -group "$_session_group_1" { {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.a_in} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.b_in} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.result} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.mul_fix_out} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.zero_check} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.M_result} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.e_result0} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.e_result} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.overflow} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.sign} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.mul.overflow_mask} }

set _session_group_2 Group2
gui_sg_create "$_session_group_2"
set Group2 "$_session_group_2"

gui_sg_addsignal -group "$_session_group_2" { {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.a_operand} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.b_operand} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.result} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.Exception} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.Comp_enable} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.output_sign} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.operand_a} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.operand_b} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.significand_a} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.significand_b} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.exponent_diff} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.significand_b_add} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.exponent_b_add} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.significand_add} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.add_sum} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.operation_sub_addBar} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.exp_a} {tb.DUT.gen_sa_row[3].gen_sa_col[3].genblk1.genblk1.PE_ij.add.exp_b} }

# Global: Highlighting

# Global: Stack
gui_change_stack_mode -mode list

# Post database loading setting...

# Restore C1 time
gui_set_time -C1_only 10049838



# Save global setting...

# Wave/List view global setting
gui_cov_show_value -switch false

# Close all empty TopLevel windows
foreach __top [gui_ekki_get_window_ids -type TopLevel] {
    if { [llength [gui_ekki_get_window_ids -parent $__top]] == 0} {
        gui_close_window -window $__top
    }
}
gui_set_loading_session_type noSession
# DVE View/pane content session: 


# View 'Wave.2'
gui_wv_sync -id ${Wave.2} -switch false
set groupExD [gui_get_pref_value -category Wave -key exclusiveSG]
gui_set_pref_value -category Wave -key exclusiveSG -value {false}
set origWaveHeight [gui_get_pref_value -category Wave -key waveRowHeight]
gui_list_set_height -id Wave -height 25
set origGroupCreationState [gui_list_create_group_when_add -wave]
gui_list_create_group_when_add -wave -disable
gui_marker_set_ref -id ${Wave.2}  C1
gui_wv_zoom_timerange -id ${Wave.2} 9891995 10207702
gui_list_add_group -id ${Wave.2} -after {New Group} {Group2}
gui_seek_criteria -id ${Wave.2} {Any Edge}



gui_set_env TOGGLE::DEFAULT_WAVE_WINDOW ${Wave.2}
gui_set_pref_value -category Wave -key exclusiveSG -value $groupExD
gui_list_set_height -id Wave -height $origWaveHeight
if {$origGroupCreationState} {
	gui_list_create_group_when_add -wave -enable
}
if { $groupExD } {
 gui_msg_report -code DVWW028
}
gui_list_set_filter -id ${Wave.2} -list { {Buffer 1} {Input 1} {Others 1} {Linkage 1} {Output 1} {Parameter 1} {All 1} {Aggregate 1} {LibBaseMember 1} {Event 1} {Assertion 1} {Constant 1} {Interface 1} {BaseMembers 1} {Signal 1} {$unit 1} {Inout 1} {Variable 1} }
gui_list_set_filter -id ${Wave.2} -text {*}
gui_list_set_insertion_bar  -id ${Wave.2} -group Group2  -position in

gui_marker_move -id ${Wave.2} {C1} 10049838
gui_view_scroll -id ${Wave.2} -vertical -set 0
gui_show_grid -id ${Wave.2} -enable false
# Restore toplevel window zorder
# The toplevel window could be closed if it has no view/pane
if {[gui_exist_window -window ${TopLevel.3}]} {
	gui_set_active_window -window ${TopLevel.3}
	gui_set_active_window -window ${Wave.2}
}
#</Session>

