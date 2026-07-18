# OJO OJO CORRER ESTO DESPUES DE OBTENER EL DATASET FINAL
# "expected" DEBE CALCULARSE CON EPITOOLS


# ------------------------------------------------------------
# 0) LIBRERÍAS
# ------------------------------------------------------------
library(dplyr)
library(sf)
library(INLA)
library(spdep)

# ------------------------------------------------------------
# 1) DATOS (SIN AGREGAR)
# ------------------------------------------------------------
obs_exp <- read.csv("D:/clases/UDES/fortalecimiento institucional/macroproyecto_2025/leish/ci/data_final_15_sep.csv")

obs_exp$DANE <- as.character(obs_exp$DANE)
obs_exp$period <- as.character(obs_exp$Year) 

# ------------------------------------------------------------
# 2) SHAPEFILE Y GRAFO (ORIGINAL)
# ------------------------------------------------------------
shp <- st_read("D:/clases/UDES/MAPAS PROYECTOS/mapa municipios/MGN_MPIO_POLITICO_wgs84_sin_San_Andres.shp",
               quiet = TRUE)

shp <- st_make_valid(shp)
shp$DANE <- as.character(shp$DANE)

# ORDENAR (CRÍTICO)
shp <- shp %>% arrange(DANE)

# ------------------------------------------------------------
# 3) HACER CONSISTENTE SHAPEFILE Y DATOS
# ------------------------------------------------------------

# Intersección válida
dane_validos <- intersect(shp$DANE, unique(obs_exp$DANE))
cat("Municipios válidos:", length(dane_validos), "\n")

# Filtrar shapefile
shp2 <- shp %>%
  filter(DANE %in% dane_validos) %>%
  arrange(DANE)

nb <- poly2nb(shp2, queen = TRUE)

adj_file <- tempfile(fileext = ".adj")
nb2INLA(adj_file, nb)
g <- inla.read.graph(adj_file)

# Crear índice espacial consistente
shp2 <- shp2 %>%
  mutate(idx_espacial = row_number())

# Filtrar datos
obs_exp_filtrado <- obs_exp %>%
  filter(DANE %in% dane_validos)

# ------------------------------------------------------------
# 4) CONSTRUIR DATASET MUNICIPIO–PERIODO
# ------------------------------------------------------------
datos_modelo <- obs_exp_filtrado %>%
  left_join(
    shp2 %>% st_drop_geometry() %>% select(DANE, idx_espacial),
    by = "DANE"
  ) %>%
  arrange(DANE, period)

cat("NAs en idx_espacial:", sum(is.na(datos_modelo$idx_espacial)), "\n")

stopifnot(sum(is.na(datos_modelo$idx_espacial)) == 0)
stopifnot(length(unique(datos_modelo$idx_espacial)) == g$n)

# ------------------------------------------------------------
# 5) ÍNDICES TEMPORALES E INTERACCIÓN
# ------------------------------------------------------------

# Índice temporal
datos_modelo <- datos_modelo %>%
  mutate(
    idx_tiempo = as.numeric(as.factor(period))
  )

# Índice IID espacial
datos_modelo <- datos_modelo %>%
  mutate(
    idx_espacial_iid = idx_espacial
  )

# Interacción espacio-tiempo
datos_modelo <- datos_modelo %>%
  mutate(
    idx_interaccion = interaction(idx_espacial, idx_tiempo, drop = TRUE) %>%
      as.numeric()
  )

# Offset
datos_modelo <- datos_modelo %>%
  mutate(
    log_E = log(expected + 0.001)
  )

# Resumen
cat("Filas totales:", nrow(datos_modelo), "\n")
cat("Municipios:", length(unique(datos_modelo$idx_espacial)), "\n")
cat("Periodos:", length(unique(datos_modelo$idx_tiempo)), "\n")

# ------------------------------------------------------------
# 6) MODELO ESPACIO-TEMPORAL (BYM + TIEMPO)
# ------------------------------------------------------------

formula_st <- cases ~ 1 +
  
  # 🔹 Espacial estructurado (ICAR)
  f(idx_espacial,
    model       = "besag",
    graph       = g,
    scale.model = TRUE,
    hyper       = list(
      prec = list(prior = "loggamma", param = c(1, 0.01))
    )
  ) +
  
  # 🔹 Espacial no estructurado
  f(idx_espacial_iid,
    model = "iid",
    hyper = list(
      prec = list(prior = "loggamma", param = c(1, 0.01))
    )
  ) +
  
  # 🔹 Temporal (RW1)
  f(idx_tiempo,
    model = "rw1",
    hyper = list(
      prec = list(prior = "loggamma", param = c(1, 0.01))
    )
  ) +
  
  # 🔹 Interacción espacio-tiempo
  f(idx_interaccion,
    model = "iid",
    hyper = list(
      prec = list(prior = "loggamma", param = c(1, 0.01))
    )
  )

# ------------------------------------------------------------
# 7) AJUSTE DEL MODELO
# ------------------------------------------------------------

fit_st <- inla(
  formula           = formula_st,
  family            = "nbinomial",
  data              = datos_modelo,
  offset            = log_E,
  control.predictor = list(compute = TRUE),
  control.compute   = list(
    dic    = TRUE,
    waic   = TRUE,
    config = TRUE
  ),
  verbose = FALSE
)

summary(fit_st)


# ------------------------------------------------------------
# 7) SIR POSTERIOR
# ------------------------------------------------------------

lp <- fit_st$summary.linear.predictor

# Verifique nombres de columnas
print(names(lp))

# SIR = exp(eta) = exp(log(mu) - log(E))
datos_modelo$SIR_mean  <- exp(lp$mean - datos_modelo$log_E)
datos_modelo$SIR_lwr95 <- exp(lp$`0.025quant` - datos_modelo$log_E)
datos_modelo$SIR_upr95 <- exp(lp$`0.975quant` - datos_modelo$log_E)

# Probabilidad de exceso (aproximación)
datos_modelo$excess <- as.integer(datos_modelo$SIR_lwr95 > 1)

# Resumen
summary(datos_modelo$SIR_mean)

str(datos_modelo)



datos_modelo <- datos_modelo %>%
  # 1. Ordenar por municipio y año para asegurar la secuencia temporal
  arrange(DANE, Year) %>%
  
  # 2. Agrupar por el código DANE de cada municipio
  group_by(DANE) %>%
  
  # 3. Crear la nueva variable usando lead() para obtener el valor del año siguiente
  mutate(excess_tp1 = lead(excess, n = 1)) %>%
  
  # 4. Quitar el agrupamiento para evitar problemas en operaciones futuras
  ungroup()

write.csv(datos_modelo, "D:/clases/UDES/fortalecimiento institucional/macroproyecto_2025/leish/ci/data_final_15_sep.csv")
