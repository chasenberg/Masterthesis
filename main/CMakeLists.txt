# Executables
add_executable(Bd2JpsiKSVariablesGrimReaper Bd2JpsiKSVariablesGrimReaper.cpp)
#add_executable(PID_efficiences PID_efficiences.cpp)
#add_executable(Bs2DstD_cuts Bs2DstD_cuts.cpp)


# Aliases for module libs

# Linker informationa
target_link_libraries(Bd2JpsiKSVariablesGrimReaper ${ALL_LIBRARIES})
#target_link_libraries(PID_efficiences ${ALL_LIBRARIES})
#target_link_libraries(Bs02DD_cuts ${ALL_LIBRARIES})



