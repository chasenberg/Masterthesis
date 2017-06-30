# Executables
add_executable(Bd2JpsiKSVariablesGrimReaper Bd2JpsiKSVariablesGrimReaper.cpp)
#add_executable(hello hello.cpp)
#add_executable(WrongPVReducer WrongPVReducer.cpp)
#add_executable(CandidateSelectionGrimReaper  CandidateSelectionGrimReaper.cpp)


# Aliases for module libs

# Linker informationa
target_link_libraries(Bd2JpsiKSVariablesGrimReaper ${ALL_LIBRARIES})
#target_link_libraries(hello ${ALL_LIBRARIES})
#target_link_libraries(WrongPVReducer ${ALL_LIBRARIES})
#target_link_libraries(CandidateSelectionGrimReaper ${ALL_LIBRARIES})


