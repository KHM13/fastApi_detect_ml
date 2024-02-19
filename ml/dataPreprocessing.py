import dask.dataframe as dd
import joblib
import pandas as pd
import numpy as np
import traceback
from os import path
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Normalizer, MaxAbsScaler

# 데이터 타입 변환
from config.common import BASE_DIR


def data_type_control2(df, column, data_type):
    # print(f"{column} : {df[column].dtype}")
    try:
        if data_type != str(df[column].dtype):
            if data_type in "datetime":
                df[column] = dd.to_datetime(df[column])
            elif data_type in "category":
                df[column] = df[column].factorize()[0]
                # df = df.astype({column: data_type})
                df[column] = df[column].astype(data_type)
            else:
                # df = df.astype({column: data_type})
                df[column] = df[column].astype(data_type)
        # print(f"{column} :: {df[column].dtype}")
    except:
        print(traceback.print_exc())
    return df


# 데이터 정규화
def process_scaler2(df, column, process):
    if process == "standard":
        scaler = StandardScaler()
    elif process == "robust":
        scaler = RobustScaler()
    elif process == "minmax":
        scaler = MinMaxScaler()
    elif process == "normal":
        scaler = Normalizer()
    elif process == "maxabs":
        scaler = MaxAbsScaler()

    df_scaled = scaler.fit_transform(df[column].reshape(1, -1))
    print(df_scaled.head())
    return df


# 문자열 변환
def fs_replace_value2(df, column, work_input, replace_input):
    df_data = df.copy()
    if str(work_input).__contains__("*"):
        replace_list = {}
        work_input = str(work_input).replace("*", "")
        for value in df_data[column].unique():
            re_value = str(value).replace(work_input, replace_input)
            replace_list[value] = re_value
        for key in replace_list.keys():
            df_data = df_data.replace({column: key}, replace_list[key])
    else:
        df_data = df_data.replace({column: work_input}, replace_input)
    return df_data


def get_remove_list():
    remove_list = ['TR_DTM', 'EBNK_MED_DSC', 'NBNK_C', 'E_FNC_USR_ACS_DSC', 'E_FNC_MED_SVRNM', 'E_FNC_RSP_C', 'E_FNC_TR_ACNO_C', 'IO_EA_PW_CD_DS1', 'IO_EA_PW_CD_DS3', 'EXCEPTION_ADD_AUTHEN_YN',
                   'SMART_AUTHEN_YN', 'ATTC_DS', 'doaddress', 'pc_PrxyCntryCd', 'pc_VpnCntryCd', 'pc_FORGERY_MAC_YN', 'pc_FORGERY_MAC_ETH0_YN', 'pc_FORGERY_MAC_ETH1_YN', 'pc_FORGERY_MAC_ETH2_YN',
                   'pc_FORGERY_MAC_ETH3_YN', 'pc_FORGERY_MAC_ETH4_YN', 'pc_FORGERY_MAC_ETH5_YN', 'pc_isVm', 'pc_vmName', 'pc_SCAN_CNT_DETECT', 'pc_SCAN_CNT_CURED', 'pc_os', 'pc_OsRemoteYn',
                   'pc_REMOTE_YN', 'pc_RemoteProg', 'pc_RemotePORT', 'pc_remoteInfo2', 'pc_remoteInfo3', 'pc_remoteInfo4', 'pc_remoteInfo5', 'pc_remoteInfo6', 'pc_remoteInfo7', 'pc_remoteInfo8',
                   'sm_locale', 'sm_roaming', 'sm_wifiApSsid', 'isForeigner']
    return remove_list


# 전처리 자동 최적화
def fs_preprocess_optimization2(df_data, target):

    intList = ['E_FNC_LGIN_DSC', 'Amount', 'IO_EA_DD1_FTR_LMT3', 'IO_EA_TM1_FTR_LMT3', 'IO_EA_DPZ_PL_IMP_BAC', 'IO_EA_TOT_BAC6', 'IO_EA_RMT_FEE1', 'totalScore']
    ynList = ['EXE_YN', 'PRE_ASSIGN_YN', 'EXCEPT_REGIST', 'pc_isVpn', 'isNewAccount', 'isNewDevice']

    try:
        # 결측치 처리
        missing_count = ((df_data.isna().sum() / df_data.index.size) * 100).compute()
        # print(f"missing_count : {missing_count}")

        df_data = df_data.replace('', np.nan)
        df_data = df_data.replace('NULL', np.nan)
        df_data = df_data.replace('null', np.nan)
        df_data = df_data.replace('Null', np.nan)

        df_data = df_data.fillna(0).persist()
        missing_count = ((df_data.isna().sum() / df_data.index.size) * 100).compute()
        print(f"missing_count after fillna : {missing_count}")

        # 유일값 제거
        for col, value in df_data.items():
            # print(f"unique count {col} : {df_data[col].nunique().compute()}")
            unique_value = df_data[col].nunique().compute()
            if unique_value == 1:     # 데이터가 없는 경우 ( 결측치 뿐인 변수 )
                print(f"{col} value count : {unique_value}")
                df_data = df_data.drop(col, axis=1).persist()

        print(df_data.shape)

        # target 변환
        if target == "processState":
            replace_condition = {"FRAUD": "1", "OVERLAPFRAUD": "1", "ACC": "1"}
            
            df_data = df_data.replace({target: replace_condition})
            target_values = [v for v in df_data[target].unique() if v != "1"]
            for value in target_values:
                df_data = fs_replace_value2(df_data, target, value, "0")
            df_data[target] = df_data[target].astype(int)

            # print(f"{target} 변환 후 : {df_data[target].nunique().compute()}")
        
        # 불필요 변수 제거
        remove_list = get_remove_list()
        if len(remove_list) > 0:
            for remove_col in remove_list:
                df_data = df_data.drop(remove_col, axis=1).persist()
        
        # 숫자형 데이터 정규화
        if len(intList) > 0:
            for int_col in intList:
                df_data = data_type_control2(df_data, int_col, "int")
                df_data = process_scaler2(df_data, int_col, "robust")

        # Y/N 데이터 변환
        if len(ynList) > 0:
            for yn_col in ynList:
                replace_dict = {"y": "1", "Y": "1", "n": "0", "N": "0"}
                df_data = df_data.replace({yn_col: replace_dict})
                df_data = data_type_control2(df_data, yn_col, "int")

        for col, value in df_data.items():
            print(f"{col} : {df_data[col].value_counts().compute()}")

    except:
        trace_back_message = traceback.format_exc()
        print(trace_back_message)
    return df_data



# 문자열 변환
def fs_replace_value(df, column, work_input, replace_input):
    df_data = df.copy()
    if str(work_input).__contains__("*"):
        replace_list = {}
        work_input = str(work_input).replace("*", "")
        for value in df_data[column].unique():
            re_value = str(value).replace(work_input, replace_input)
            replace_list[value] = re_value
        for key in replace_list.keys():
            df_data = df_data.replace({column: key}, replace_list[key])
    else:
        df_data = df_data.replace({column: work_input}, replace_input)
    return df_data


def fs_preprocess_optimization(df_data: pd.DataFrame, target: str):
    try:
        # 결측치 처리
        nullList = df_data.isnull().sum()
        for col, count in nullList.items():
            df_data[col].replace('', np.nan, inplace=True)
            df_data[col].replace('NULL', np.nan, inplace=True)
            df_data[col].replace('null', np.nan, inplace=True)
            df_data[col].replace('Null', np.nan, inplace=True)

            df_data[col] = df_data[col].fillna('0')
            if len(df_data[col].unique()) == 1:     # 데이터가 없는 경우 ( 결측치 뿐인 변수 )
                if col == target: # target 변수일경우 통과
                    continue
                df_data = df_data.drop([col], axis=1)
            else:
                try:
                    df_data[col] = df_data[col].astype("float")
                except:
                    continue

        # 불필요 변수 제거
        remove_list = get_remove_list()
        if len(remove_list) > 0:
            for remove_col in remove_list:
                if remove_col in df_data.columns:
                    df_data = df_data.drop(remove_col, axis=1)

        if target == "processState":
            replace_condition = {"FRAUD": "1", "OVERLAPFRAUD": "1", "ACC": "1"}
            df_data = df_data.replace({target: replace_condition})
            target_values = [v for v in df_data[target].unique() if v != "1"]
            for value in target_values:
                df_data = fs_replace_value(df_data, target, value, "0")
            df_data[target] = df_data[target].astype("int")

        # 문자열 변환
        str_data = df_data.select_dtypes(include="object")
        for col in str_data:
            # YN 변환
            try:
                replace_dict = {"y": "1", "Y": "1", "n": "0", "N": "0"}
                df_data = df_data.replace({col: replace_dict})
                df_data[col] = df_data[col].astype("int")
                continue
            except:
                pass

            try:
                encoder = LabelEncoder()
                df_data[col] = encoder.fit_transform(df_data[col])
            except:
                df_data.loc[:, [col]] = df_data.loc[:, [col]].astype("float")
                encoder = LabelEncoder()
                df_data[col] = encoder.fit_transform(df_data[col])
            df_data.loc[:, [col]] = df_data.loc[:, [col]].astype("int")
    except:
        trace_back_message = traceback.format_exc()
        print(trace_back_message)
    return df_data


# 전처리 자동 최적화
def fs_preprocess_encoder(df_data: pd.DataFrame, target: str):
    try:
        # 문자열 변환
        str_data = df_data.select_dtypes(include="object")
        # print(f"문자열 컬럼 : {str_data}")
        for col in str_data:
            try:
                encoder = LabelEncoder()
                df_data[col] = encoder.fit_transform(df_data[col])
            except:
                df_data.loc[:, [col]] = df_data.loc[:, [col]].astype("float")
                encoder = LabelEncoder()
                df_data[col] = encoder.fit_transform(df_data[col])
            df_data.loc[:, [col]] = df_data.loc[:, [col]].astype("int")

        # print(df_data.head())
        # print(df_data.shape)
    except:
        trace_back_message = traceback.format_exc()
        print(trace_back_message)
    return df_data


# 기본 데이터 전처리
def fs_preprocess_optimization_for_data(df_data: pd.DataFrame, target: str):
    try:
        # 결측치 처리
        nullList = df_data.isnull().sum()
        for col, count in nullList.items():
            df_data[col].replace('', np.nan, inplace=True)
            df_data[col].replace('NULL', np.nan, inplace=True)
            df_data[col].replace('null', np.nan, inplace=True)
            df_data[col].replace('Null', np.nan, inplace=True)

            df_data[col] = df_data[col].fillna('0')
            if len(df_data[col].unique()) == 1:     # 데이터가 없는 경우 ( 결측치 뿐인 변수 )
                if col == target: # target 변수일경우 통과
                    continue
                df_data = df_data.drop([col], axis=1)
            else:
                try:
                    df_data[col] = df_data[col].astype("float")
                except:
                    continue

        # 불필요 변수 제거
        remove_list = get_remove_list()
        if len(remove_list) > 0:
            for remove_col in remove_list:
                if remove_col in df_data.columns:
                    df_data = df_data.drop(remove_col, axis=1)

        if target == "processState":
            replace_condition = {"FRAUD": "1", "OVERLAPFRAUD": "1", "ACC": "1"}
            df_data = df_data.replace({target: replace_condition})
            target_values = [v for v in df_data[target].unique() if v != "1"]
            for value in target_values:
                df_data = fs_replace_value(df_data, target, value, "0")
            df_data[target] = df_data[target].astype("int")

        # 문자열 변환
        str_data = df_data.select_dtypes(include="object")
        for col in str_data:
            # YN 변환
            try:
                replace_dict = {"y": "1", "Y": "1", "n": "0", "N": "0"}
                df_data = df_data.replace({col: replace_dict})
                df_data[col] = df_data[col].astype("int")
            except:
                continue
    except:
        trace_back_message = traceback.format_exc()
        print(trace_back_message)
    return df_data



def fs_preprocess_optimization_for_npas(df_data: pd.DataFrame, target: str):
    try:
        # 결측치 처리
        nullList = df_data.isnull().sum()
        for col, count in nullList.items():
            df_data[col].replace('', np.nan, inplace=True)
            df_data[col].replace('NULL', np.nan, inplace=True)
            df_data[col].replace('null', np.nan, inplace=True)
            df_data[col].replace('Null', np.nan, inplace=True)

            df_data[col] = df_data[col].fillna('0')
            if len(df_data[col].unique()) == 1:     # 데이터가 없는 경우 ( 결측치 뿐인 변수 )
                if col == target: # target 변수일경우 통과
                    continue
                df_data = df_data.drop([col], axis=1)
            else:
                try:
                    df_data[col] = df_data[col].astype("float")
                except:
                    continue

        # 불필요 변수 제거
        remove_list = ['거래일시', 'clientID', '탐지결과']
        if len(remove_list) > 0:
            for remove_col in remove_list:
                if remove_col in df_data.columns:
                    df_data = df_data.drop(remove_col, axis=1)

        if target == "mlResult":
            condition = ["Googlebot", "bingbot", "Yeti", "Daumoa", "Zumbot", "Chrome", "Chrome Mobile", "Edge", "Firefox", "IE",
                         "Mobile Safari", "Safari", "Samsung Internet", "Whale", "Opera"]
            
            def func(x):
                if x in condition:
                    return "0"
                else:
                    return "1"
            df_data[target] = df_data["client_userAgentName"].apply(lambda x : func(x))
            df_data[target] = df_data[target].astype("int")

        columns: list = df_data.columns.to_list()

        # 문자열 변환
        str_data = df_data.select_dtypes(include="object")
        for col in str_data:
            # YN 변환
            try:
                replace_dict = {"y": "1", "Y": "1", "n": "0", "N": "0"}
                df_data = df_data.replace({col: replace_dict})
                df_data[col] = df_data[col].astype("int")
                continue
            except:
                pass

        encoder = joblib.load(f'{BASE_DIR}/dask/output/encoder.pkl') if path.exists(
            f'{BASE_DIR}/dask/output/encoder.pkl') else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=9999)
        encoded_data = encoder.fit_transform(df_data)
        columns = df_data.columns
        df_data = pd.DataFrame(encoded_data, columns=columns)
        # print(encoder.categories_)
        # print(df_data)
        joblib.dump(encoder, f'{BASE_DIR}/dask/output/encoder.pkl')
    except:
        trace_back_message = traceback.format_exc()
        print(trace_back_message)
    return df_data
