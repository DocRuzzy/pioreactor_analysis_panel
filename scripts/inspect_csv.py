import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pioreactor_analysis.core.csv_parser import PioreactorCSVParser

p = r'c:/Development/python-projects/Pioreactor/pioreactor_analysis_panel/data/export_20260109190754_p2/pioreactor_unit_activity_data_rollup/pioreactor_unit_activity_data_rollup-maxtest-pioreactor2-20260109140754.csv'
parser = PioreactorCSVParser()

try:
    data = parser.parse(p)
    print('Parsed type:', type(data).__name__)
    if hasattr(data, 'to_dataframe'):
        df = data.to_dataframe()
    elif hasattr(data, 'to_od_dataframe'):
        df = data.to_od_dataframe()
    else:
        df = None

    print('Columns:', list(df.columns) if df is not None else None)
    print('\nFirst 10 rows:')
    if df is not None:
        print(df.head(10).to_csv(index=False))
    else:
        print('No dataframe produced')
except Exception as e:
    import traceback
    print('Parser error:', e)
    traceback.print_exc()
