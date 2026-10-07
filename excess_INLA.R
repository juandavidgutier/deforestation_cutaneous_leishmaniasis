# WARNING WARNING RUN THIS AFTER OBTAINING THE FINAL DATASET
# "expected" MUST BE COMPUTED WITH EPITOOLS


# ------------------------------------------------------------
# 0) LIBRARIES
# ------------------------------------------------------------
library(dplyr)
library(sf)
library(INLA)
library(spdep)

# ------------------------------------------------------------
# 1) DATA (NOT AGGREGATED)
# ------------------------------------------------------------
obs_exp <- read.csv("D:/data.csv")

obs_exp$DANE <- as.character(obs_exp$DANE)
obs_exp$period <- as.character(obs_exp$Year) 

# ------------------------------------------------------------
# 2) SHAPEFILE AND GRAPH (ORIGINAL)
# ------------------------------------------------------------
shp <- st_read("D:/map/MGN_MPIO_POLITICO_wgs84_sin_San_Andres.shp",
               quiet = TRUE)

shp <- st_make_valid(shp)
shp$DANE <- as.character(shp$DANE)

# SORT (CRITICAL)
shp <- shp %>% arrange(DANE)

# ------------------------------------------------------------
# 3) MAKE SHAPEFILE AND DATA CONSISTENT
# ------------------------------------------------------------

# Valid intersection
dane_validos <- intersect(shp$DANE, unique(obs_exp$DANE))
cat("Valid municipalities:", length(dane_validos), "\n")

# Filter shapefile
shp2 <- shp %>%
  filter(DANE %in% dane_validos) %>%
  arrange(DANE)

nb <- poly2nb(shp2, queen = TRUE)

adj_file <- tempfile(fileext = ".adj")
nb2INLA(adj_file, nb)
g <- inla.read.graph(adj_file)

# Create consistent spatial index
shp2 <- shp2 %>%
  mutate(idx_espacial = row_number())

# Filter data
obs_exp_filtrado <- obs_exp %>%
  filter(DANE %in% dane_validos)

# ------------------------------------------------------------
# 4) BUILD MUNICIPALITY–PERIOD DATASET
# ------------------------------------------------------------
datos_modelo <- obs_exp_filtrado %>%
  left_join(
    shp2 %>% st_drop_geometry() %>% select(DANE, idx_espacial),
    by = "DANE"
  ) %>%
  arrange(DANE, period)

cat("NAs in idx_espacial:", sum(is.na(datos_modelo$idx_espacial)), "\n")

stopifnot(sum(is.na(datos_modelo$idx_espacial)) == 0)
stopifnot(length(unique(datos_modelo$idx_espacial)) == g$n)

# ------------------------------------------------------------
# 5) TEMPORAL INDICES AND INTERACTION
# ------------------------------------------------------------

# Temporal index
datos_modelo <- datos_modelo %>%
  mutate(
    idx_tiempo = as.numeric(as.factor(period))
  )

# Spatial IID index
datos_modelo <- datos_modelo %>%
  mutate(
    idx_espacial_iid = idx_espacial
  )

# Space-time interaction
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

# Summary
cat("Total rows:", nrow(datos_modelo), "\n")
cat("Municipalities:", length(unique(datos_modelo$idx_espacial)), "\n")
cat("Periods:", length(unique(datos_modelo$idx_tiempo)), "\n")

# ------------------------------------------------------------
# 6) SPACE-TIME MODEL (BYM + TIME)
# ------------------------------------------------------------

formula_st <- cases ~ 1 +
  
  # 🔹 Structured spatial (ICAR)
  f(idx_espacial,
    model       = "besag",
    graph       = g,
    scale.model = TRUE,
    hyper       = list(
      prec = list(prior = "loggamma", param = c(1, 0.01))
    )
  ) +
  
  # 🔹 Unstructured spatial
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
  
  # 🔹 Space-time interaction
  f(idx_interaccion,
    model = "iid",
    hyper = list(
      prec = list(prior = "loggamma", param = c(1, 0.01))
    )
  )

# ------------------------------------------------------------
# 7) MODEL FITTING
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

# Check column names
print(names(lp))

# SIR = exp(eta) = exp(log(mu) - log(E))
datos_modelo$SIR_mean  <- exp(lp$mean - datos_modelo$log_E)
datos_modelo$SIR_lwr95 <- exp(lp$`0.025quant` - datos_modelo$log_E)
datos_modelo$SIR_upr95 <- exp(lp$`0.975quant` - datos_modelo$log_E)

# Excess probability (approximation)
datos_modelo$excess <- as.integer(datos_modelo$SIR_lwr95 > 1)

# Summary
summary(datos_modelo$SIR_mean)

str(datos_modelo)



datos_modelo <- datos_modelo %>%
  # 1. Sort by municipality and year to ensure the temporal sequence
  arrange(DANE, Year) %>%
  
  # 2. Group by the DANE code of each municipality
  group_by(DANE) %>%
  
  # 3. Create the new variable using lead() to obtain the next year's value
  mutate(excess_tp1 = lead(excess, n = 1)) %>%
  
  # 4. Remove the grouping to avoid problems in future operations
  ungroup()

write.csv(datos_modelo, "D:/data.csv")
