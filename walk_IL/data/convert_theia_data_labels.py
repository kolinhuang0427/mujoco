import numpy as np
import pandas as pd
import pickle

def convert_theia_to_myo_data(theia_table_path, output_path):
    record_time_sec = 30
    df = pd.read_csv(theia_table_path, skiprows=lambda x: x in range(0, record_time_sec*1000 + 8), encoding_errors='ignore', on_bad_lines='skip', low_memory=False)
    def clean_column_name(col, idx):
        if isinstance(col, str) and col.startswith('Unnamed:'):
            if idx > 0 and isinstance(df.columns[idx-1], str) and ":" in df.columns[idx-1] and not df.columns[idx-1].startswith('Unnamed:'):
                curr = df.columns[idx-1].split(":")[1].strip()
                result = curr + " " + str(df.iloc[0, idx]) + " " + str(df.iloc[1, idx])
                return result
            elif idx > 1 and isinstance(df.columns[idx-2], str) and ":" in df.columns[idx-2] and not df.columns[idx-2].startswith('Unnamed:'):
                curr = df.columns[idx-2].split(":")[1].strip() 
                result = curr + " " + str(df.iloc[0, idx]) + " " + str(df.iloc[1, idx])
                return result
            else:
                val0 = str(df.iloc[0, idx]) if pd.notna(df.iloc[0, idx]) else "Col"
                val1 = str(df.iloc[1, idx]) if pd.notna(df.iloc[1, idx]) else "Unit"
                result = f"Unknown_{val0}_{val1}_{idx}"
                return result
        elif isinstance(col, str) and ":" in col:
            curr = col.split(":")[1].strip()
            result = curr + " " + str(df.iloc[0, idx]) + " " + str(df.iloc[1, idx])
            return result
        else:
            return col

    df.columns = [clean_column_name(col, idx) for idx, col in enumerate(df.columns)]

    df = df[2:].reset_index(drop=True)
    
    for col in df.columns:
        if isinstance(col, str) and "deg" in col:
            df[col] = df[col].astype(float) * 0.0174533
        if isinstance(col, str) and "mm" in col:
            df[col] = df[col].astype(float) * 0.001
    myo_data = {'qpos': pd.DataFrame()}

    def modify(myo_var, theia_var, flip = 1):
        myo_data['qpos'][myo_var] = flip * df[theia_var]

    modify("hip_flexion_l", "LeftHipAngles_Theia X deg")
    modify("hip_flexion_r", "RightHipAngles_Theia X deg")

    modify("hip_adduction_l", "LeftHipAngles_Theia Y deg")
    modify("hip_adduction_r", "RightHipAngles_Theia Y deg")

    modify("hip_rotation_l", "LeftHipAngles_Theia Z deg")
    modify("hip_rotation_r", "RightHipAngles_Theia Z deg")

    modify("knee_angle_l", "LeftKneeAngles_Theia X deg")
    modify("knee_angle_r", "RightKneeAngles_Theia X deg")

    modify("ankle_angle_l", "LeftAnkleAngles_Theia X deg")
    modify("ankle_angle_r", "RightAnkleAngles_Theia X deg")

    modify("mtp_angle_l", "l_toes_4X4 RX deg", -1)
    modify("mtp_angle_r", "r_toes_4X4 RX deg", -1)

    modify("subtalar_angle_l", "LeftAnkleAngles_Theia Y deg")
    modify("subtalar_angle_r", "RightAnkleAngles_Theia Y deg")

    with open(output_path, 'wb') as file:
        pickle.dump(myo_data, file)
    print(f"Done! {output_path} saved")
    
    # Also save as CSV files
    # base_path = output_path.rsplit('.', 1)[0]  # Remove file extension
    # myo_data['qpos'].to_csv(f"{base_path}_qpos.csv", index=False)

if __name__ == '__main__':
    convert_theia_to_myo_data('AB01_Jimin_1p0mps_1.csv', 'expert_data.pkl')