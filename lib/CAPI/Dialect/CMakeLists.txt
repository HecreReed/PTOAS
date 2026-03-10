add_mlir_public_c_api_library(PTOCAPI
    PTO.cpp

    LINK_LIBS PUBLIC
    PTOIR
    MLIRCAPIIR
)
