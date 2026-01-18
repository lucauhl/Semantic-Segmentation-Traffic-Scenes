import osmnx as ox
import pandas as pd
from typing import Optional, Tuple

# Funktion zur Verarbeitung einer einzelnen Region
def process_region(region_name):
    print(f"Starte Verarbeitung für: {region_name}")
    try:
        G = ox.graph_from_place(region_name, network_type='drive')

        # Projizieren und bereinigen
        # dead_ends=False entfernt bereits Endpunkte (1 Straße), aber behält 3-Wege-Kreuzungen
        G_proj = ox.project_graph(G)
        G_clean = ox.consolidate_intersections(G_proj, tolerance=15, rebuild_graph=True, dead_ends=False)

        # Nodes extrahieren
        nodes_gdf, _ = ox.graph_to_gdfs(G_clean)

        # Das Attribut 'street_count' gibt an, wie viele Straßenarme von diesem Knoten abgehen.
        # Wir filtern alles heraus, was nicht genau 4 ist (also z.B. T-Kreuzungen mit 3).
        if 'street_count' in nodes_gdf.columns:
            nodes_gdf = nodes_gdf[nodes_gdf['street_count'] == 4].copy()
        else:
            print(f"Warnung: 'street_count' Spalte fehlt bei {region_name}. Filterung nicht möglich.")

        # Sicherstellen, dass es WGS84 ist
        if nodes_gdf.crs.to_epsg() != 4326:
            nodes_gdf = nodes_gdf.to_crs(epsg=4326)

        # GeoDataFrame NICHT in ein normales DataFrame umwandeln,
        # sonst gehen Geo-Attribute (x/y) verloren und es kommt zu Fehlern
        df_export = nodes_gdf.copy()

        print(f" -> {len(df_export)} echte 4-Wege-Kreuzungen in {region_name} gefunden.")
        return df_export

    except Exception as e:
        print(f"!!! Fehler bei {region_name}: {e}")
        return None

"""
admin_level=2: Staatsgrenze (Deutschland)

admin_level=4: Bundesländer (z.B. Bayern, Baden-Württemberg)

admin_level=5: Regierungsbezirke (nur in einigen Bundesländern wie BW, Bayern, Hessen, NRW)

admin_level=6: Landkreise und kreisfreie Städte (z.B. Landkreis München, Stadt Stuttgart).

admin_level=8: Städte und Gemeinden (z.B. Tegernheim)

admin_level=9/10: Stadtbezirke/Stadtteile (in Großstädten)
"""

def get_municipalities_in_state(state_name):
    print(f"--- Suche alle Gemeinden in {state_name} ---")
    try:
        # admin_level=8 ist in DE meist die politische Gemeinde/Stadt
        tags = {'admin_level': '8'}
        gdf = ox.features_from_place(state_name, tags=tags)

        if 'name' in gdf.columns:
            # dropna() entfernt Einträge ohne Namen, unique() entfernt Dopplungen
            municipality_list = gdf['name'].dropna().unique().tolist()
            print(f" -> {len(municipality_list)} Gemeinden in {state_name} gefunden.")
            return municipality_list
        else:
            print(f"Keine 'name'-Spalte in den Daten für {state_name} gefunden.")
            return []
    except Exception as e:
        print(f"Fehler beim Laden der Gemeindeliste für {state_name}: {e}")
        return []

def get_unique_cities_from_gps_data(coarse_csv_path, fine_csv_path):
    """
    Reads Cityscapes coarse and fine GPS CSV files and returns a sorted
    list of unique city names present in either dataset.
    """
    # Load the datasets
    coarse_df = pd.read_csv(coarse_csv_path)
    fine_df = pd.read_csv(fine_csv_path)

    # Extract unique cities from each dataframe
    # Using a set automatically handles uniqueness
    coarse_cities = set(coarse_df['city'].unique())
    fine_cities = set(fine_df['city'].unique())

    # Combine the sets to get all unique cities across both files
    all_unique_cities = coarse_cities.union(fine_cities)

    # Return the result as a sorted list of strings
    return sorted(list(all_unique_cities))


def append_intersections_to_csv(
    df_result: pd.DataFrame,
    city: str,
    output_file: str,
    used_region: Optional[str] = None,
    first_run: bool = False,
) -> Tuple[int, bool]:

    try:
        if df_result is None or df_result.empty:
            return 0, first_run

        # Ensure WGS84 if CRS info is available
        if hasattr(df_result, 'crs') and getattr(df_result, 'crs', None) is not None:
            try:
                # Only convert if not already EPSG:4326
                if df_result.crs.to_epsg() != 4326:
                    df_result = df_result.to_crs(epsg=4326)
            except Exception:
                # If CRS conversion fails, continue with the original (best effort)
                pass

        # Build export DataFrame
        if 'geometry' in df_result.columns:
            df_export = pd.DataFrame({
                'latitude': df_result.geometry.y,
                'longitude': df_result.geometry.x,
            })
        elif {'y', 'x'}.issubset(df_result.columns):
            df_export = pd.DataFrame({
                'latitude': df_result['y'],
                'longitude': df_result['x'],
            })
        else:
            print(f" -> Unerwartetes Format für '{city}' – keine Geometrie oder x/y Spalten. Überspringe Export.")
            return 0, first_run

        # Metadata
        df_export['city'] = city
        if used_region is not None:
            df_export['region_query'] = used_region
        df_export['intersection_id'] = df_result.index

        # Append to CSV
        df_export.to_csv(output_file, mode='a', header=first_run, index=False)

        # After first successful write, ensure header won't be written again
        new_first_run = False if first_run else first_run

        return len(df_export), new_first_run
    except Exception as e:
        print(f"!!! Fehler beim Aufbereiten/Speichern für {city}: {e}")
        return 0, first_run