import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, Tuple, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoomDataProcessor:
    """Process and clean room booking data from Excel files."""

    # Thai month mapping
    MONTH_MAP = {
        "ม.ค": 1, "มค": 1, "มกราคม": 1,
        "ก.พ": 2, "กพ": 2, "กุมภาพันธ์": 2,
        "มี.ค": 3, "มีค": 3, "มีนาคม": 3,
        "เม.ย": 4, "เมย": 4, "เมษายน": 4,
        "พ.ค": 5, "พค": 5, "พฤษภาคม": 5,
        "มิ.ย": 6, "มิย": 6, "มิถุนายน": 6,
        "ก.ค": 7, "กค": 7, "กรกฎาคม": 7,
        "ส.ค": 8, "สค": 8, "สิงหาคม": 8,
        "ก.ย": 9, "กย": 9, "กันยายน": 9,
        "ต.ค": 10, "ตค": 10, "ตุลาคม": 10,
        "พ.ย": 11, "พย": 11, "พฤศจิกายน": 11,
        "ธ.ค": 12, "ธค": 12, "ธันวาคม": 12,
    }

    # Room pricing information
    PRICE_MAP = {
        'หอประชุม': 2500,
        'อยุธยา-อาเซียน': 1000,
        'อยุธยา-ฮอลันดา': 200,
        'อยุธยา-โปรตุเกส': 250,
        'ห้องประชุม 317': 1000,
        'ห้องประชุม 1': 400,
        'ห้องประชุม 2': 650,
        'ห้องประชุมบ้านพลูหลวง': 650,
        'ห้องประชุม สนอ.': 2300,
        'ห้องประชุม 317 บน': 1000,
        'ห้องประชุม 317 ล่าง': 1000,
        'สนามฟุตบอล': 3000,
    }

    # Room seating capacity
    SEATS_MAP = {
        'หอประชุม': 700,
        'อยุธยา-อาเซียน': 30,
        'อยุธยา-ฮอลันดา': 8,
        'อยุธยา-โปรตุเกส': 12,
        'ห้องประชุม 317': 300,
        'ห้องประชุม 1': 20,
        'ห้องประชุม 2': 80,
        'ห้องประชุมบ้านพลูหลวง': 80,
        'ห้องประชุม สนอ.': 400,
        'ห้องประชุม 317 บน': 180,
        'ห้องประชุม 317 ล่าง': 120,
        'สนามฟุตบอล': 1300,
    }

    def __init__(self, data_dir: str = "../data"):
        """Initialize the data processor.

        Args:
            data_dir: Directory containing the Excel files
        """
        self.data_dir = Path(data_dir)
        self.clean_dir = self.data_dir / "clean"
        self.clean_dir.mkdir(exist_ok=True, parents=True)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing: strip column names and remove missing time values.

        Args:
            df: Raw dataframe from Excel

        Returns:
            Preprocessed dataframe
        """
        # Strip column names
        df.columns = df.columns.str.strip()
        logger.info(f"Columns after stripping: {df.columns.tolist()}")

        # Delete missing values from time column
        if "เวลา" in df.columns:
            df = df.dropna(subset=["เวลา"], axis=0)

        return df

    def calculate_duration(self, time_str: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate duration in hours from Thai time string.

        Args:
            time_str: Time string in format "HH.MM-HH.MM"

        Returns:
            Tuple of (duration_hours, start_time, end_time)
        """
        if pd.isna(time_str):
            return None, None, None

        # Clean the string: remove 'น', dots, and extra spaces
        time_str = str(time_str).replace("น.", "").replace("น", "").strip()
        times = time_str.split("-")

        if len(times) == 2:
            try:
                # Remove trailing dots and spaces before converting
                start_str = times[0].strip().rstrip(".")
                end_str = times[1].strip().rstrip(".")

                start = float(start_str)
                end = float(end_str)

                # Handle overnight times
                if end < start:
                    end += 24
                duration = np.ceil(end - start)
                return duration, start, end
            except (ValueError, ZeroDivisionError):
                return None, None, None

        return None, None, None

    def classify_event_simple(self, start_time: float, end_time: float) -> Optional[str]:
        """Classify event period based on start and end time.

        Args:
            start_time: Start time as float (e.g., 9.0 for 9:00)
            end_time: End time as float

        Returns:
            Event period classification
        """
        if pd.isna(start_time) or pd.isna(end_time):
            return None

        # Morning: 09.00-12.59
        if 8.0 <= start_time <= 13.0 and 8.30 <= end_time <= 13.0:
            return "Morning"
        # Afternoon: 13.00-17.59
        elif 13.0 <= start_time <= 18.0 and 13.0 <= end_time <= 18.0:
            return "Afternoon"
        # Night: 18.00-24.00
        elif 18.0 <= start_time <= 24.0 and 18.0 <= end_time <= 24.0:
            return "Night"
        else:
            return "All Day"

    def thai_to_datetime(self, date_str: str) -> pd.Timestamp:
        """Convert Thai date string (Buddhist Era) to pandas Timestamp.

        Args:
            date_str: Thai date string

        Returns:
            Converted pandas Timestamp
        """
        if pd.isna(date_str):
            return pd.NaT

        # Remove dots and spaces
        s = re.sub(r"[\. ]+", "", str(date_str).strip())

        # Extract day, month, year
        m = re.match(r"(\d{1,2})([^\d]+)(\d{2,4})", s)
        if not m:
            return pd.NaT

        day, mon_str, year_str = m.groups()
        day = int(day)
        mon = self.MONTH_MAP.get(mon_str)

        if mon is None:
            return pd.NaT

        year_be = int(year_str)
        if year_be < 100:
            year_be += 2500
        year_ce = year_be - 543

        return pd.Timestamp(year=year_ce, month=mon, day=day)

    def normalize_room(self, name: str) -> Optional[str]:
        """Normalize room names to standard format.

        Args:
            name: Raw room name

        Returns:
            Normalized room name
        """
        if pd.isna(name):
            return None

        # Clean string: remove dots, collapse spaces, lowercase
        s = re.sub(r'[^\wก-๙]', '', str(name)).lower()

        # Match room patterns
        if 'ห้องประชุมสนอ' in s:
            return 'ห้องประชุม สนอ.'
        if 'หอประชุม' in s:
            return 'หอประชุม'
        if 'อยุธยา' in s and 'อาเซียน' in s:
            return 'อยุธยา-อาเซียน'
        if 'อยุธยา' in s and 'ฮอลันดา' in s:
            return 'อยุธยา-ฮอลันดา'
        if 'อยุธยา' in s and 'โปรตุเกส' in s:
            return 'อยุธยา-โปรตุเกส'

        # Room 317 variations
        if '317' in s:
            has_on = 'บน' in s
            has_under = 'ล่าง' in s
            if has_on and has_under:
                return 'ห้องประชุม 317'
            if has_on:
                return 'ห้องประชุม 317 บน'
            if has_under:
                return 'ห้องประชุม 317 ล่าง'
            return 'ห้องประชุม 317'

        # Other rooms
        if re.search(r'ห้องประชุม1', s):
            return 'ห้องประชุม 1'
        if 'ห้องประชุม2' in s:
            return 'ห้องประชุม 2'
        if '70ล้าน' in s or 'พลูหลวง' in s:
            return 'ห้องประชุมบ้านพลูหลวง'
        if 'โรงอาหารเก่า' in s:
            return 'โรงยิม'
        if 'สนามหญ้าอาคาร2' in s:
            return 'สนามฟุตบอล'

        return None

    def extract_room(self, row: pd.Series, room_cols: List[str]) -> Optional[str]:
        """Extract room name from row based on room columns.

        Args:
            row: DataFrame row
            room_cols: List of room column names

        Returns:
            Extracted room name
        """
        for col in room_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip() != "":
                return val
        return None

    def process_file(self, file_path: Path, room_cols: List[str],
                    skip_rows: int = 1, sheet: int = 0) -> pd.DataFrame:
        """Process a single Excel file.

        Args:
            file_path: Path to Excel file
            room_cols: List of room column names
            skip_rows: Number of rows to skip
            sheet: Sheet number or name

        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing {file_path.name}")

        # Read Excel file
        df = pd.read_excel(file_path, header=0, skiprows=skip_rows, sheet_name=sheet)
        df = self.preprocess_dataframe(df)

        # Calculate duration
        if "เวลา" in df.columns:
            df["duration_hours"], df["start_time"], df["end_time"] = zip(
                *df["เวลา"].apply(self.calculate_duration)
            )

            # Classify event period
            df["event_period"] = df.apply(
                lambda row: self.classify_event_simple(row["start_time"], row["end_time"]),
                axis=1
            )

        # Convert Thai date to datetime
        if "ใช้ในวันที่" in df.columns:
            df["ใช้ในวันที่ (datetime)"] = df["ใช้ในวันที่"].apply(self.thai_to_datetime)

        # Extract room
        df["room"] = df.apply(lambda row: self.extract_room(row, room_cols), axis=1)

        # Keep only necessary columns
        keep_cols = ["หน่วยงานที่ขอใช้", "duration_hours", "start_time",
                    "end_time", "event_period", "ใช้ในวันที่ (datetime)", "room"]
        available_cols = [col for col in keep_cols if col in df.columns]
        df = df[available_cols]

        return df

    def process_all_files(self) -> pd.DataFrame:
        """Process all data files and combine them.

        Returns:
            Combined DataFrame with all cleaned data
        """
        all_dfs = []

        # Define file configurations
        file_configs = [
            ("ห้องประชุม ไปราชการปี 2557.xlsx",
             ["ห้องประชุม", "อยุธยา - อาเซียน", "อยุธยา-ฮอลันดา",
              "อยุธยา - โปรตุเกส", "ห้องประชุม 317", "หอประชุม"]),

            ("ห้องประชุม ไปราชการปี 2558.xlsx",
             ["ห้องประชุม  1/2", "อยุธยา - อาเซียน", "อยุธยา-ฮอลันดา",
              "อยุธยา - โปรตุเกส", "ห้องประชุม 317", "หอประชุม"]),

            # Add more file configurations as needed
        ]

        for file_name, room_cols in file_configs:
            file_path = self.data_dir / file_name
            if file_path.exists():
                df = self.process_file(file_path, room_cols)
                all_dfs.append(df)

                # Save individual cleaned file
                clean_path = self.clean_dir / f"cleaned_{file_name}"
                df.to_excel(clean_path, index=False)
                logger.info(f"Saved cleaned data to {clean_path}")

        # Combine all DataFrames
        all_data = pd.concat(all_dfs, ignore_index=True)

        # Normalize room names
        all_data['room'] = all_data['room'].apply(self.normalize_room)

        # Add price and seats mapping
        all_data['price'] = all_data['room'].map(self.PRICE_MAP)
        all_data['seats'] = all_data['room'].map(self.SEATS_MAP)

        # Rename columns to English
        all_data = all_data.rename(columns={
            'หน่วยงานที่ขอใช้': 'department',
            'ใช้ในวันที่ (datetime)': 'date'
        })

        # Convert date to date only (not datetime)
        if 'date' in all_data.columns:
            all_data['date'] = pd.to_datetime(all_data['date']).dt.date

        # Drop rows with missing values
        all_data = all_data.dropna()

        # Save combined data
        output_path = self.clean_dir / "cleaned_all_rooms.xlsx"
        all_data.to_excel(output_path, index=False)
        logger.info(f"Saved combined data to {output_path}")

        return all_data


def main():
    """Main function to run data preparation."""
    processor = RoomDataProcessor()

    # Process all files
    all_data = processor.process_all_files()

    # Print summary statistics
    print("\n=== Data Processing Summary ===")
    print("Total records: {len(all_data)}")
    print("Date range: {all_data['date'].min()} to {all_data['date'].max()}")
    print("Unique rooms: {all_data['room'].nunique()}")
    print("Unique departments: {all_data['department'].nunique()}")

    print("\nRoom distribution:")
    print(all_data['room'].value_counts())

    print("\nEvent period distribution:")
    print(all_data['event_period'].value_counts())

    return all_data


if __name__ == "__main__":
    main()