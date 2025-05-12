import csv
import json
import os
import random
import logging
from datetime import datetime, timedelta
import numpy as np
import skfuzzy as fuzz # Для Fuzzy Logic
from skfuzzy import control as ctrl # Для Fuzzy Logic

# --- Добавление пути к вашим модулям ---
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Если скрипт в корне проекта
# Или конкретные пути, если структура другая
# sys.path.append('путь/к/вашему/проекту')


# --- Импорт ваших классов ---
from control.markov_controller import MarkovController, Action as MarkovAction
# Mock-классы (можно скопировать из train_markov_model.py или импортировать, если они вынесены)
# Если они в train_markov_model.py, то можно импортировать их оттуда или перенести в отдельный mock_utils.py
# Для примера предположим, что они здесь или в импортируемом файле
class MockDataManager:
    def __init__(self):
        self.latest_data = {
            "scd41": {"co2": 400, "temperature": 20.0, "humidity": 50.0},
            "room": {"occupants": 0, "ventilated": False, "ventilation_speed": "off"},
            "bmp280": {"temperature": 20.0, "pressure": 1000.0},
            "timestamp": datetime.now().isoformat()
        }
    def update_sensor_data_from_row(self, csv_row: dict):
        self.latest_data["scd41"]["co2"] = float(csv_row['co2'])
        self.latest_data["scd41"]["temperature"] = float(csv_row['temperature'])
        self.latest_data["scd41"]["humidity"] = float(csv_row.get('humidity', 50.0)) # Добавим обработку отсутствия
        self.latest_data["room"]["occupants"] = int(csv_row['occupants'])
        action_str = csv_row.get('ventilation_action', 'off')
        self.latest_data["room"]["ventilated"] = action_str != 'off'
        self.latest_data["room"]["ventilation_speed"] = action_str
        self.latest_data["timestamp"] = csv_row['timestamp']

class MockPicoManager:
    def get_ventilation_status(self): return False
    def get_ventilation_speed(self): return "off"
    def control_ventilation(self, state, speed=None): return True

class MockPreferenceManager:
    def get_all_user_preferences(self): return {}
    def calculate_compromise_preference(self, user_ids):
        # Импорт модели CompromisePreference, если она определена
        # from preferences.models import CompromisePreference
        # return CompromisePreference(
        #     user_count=0, temp_min=20.0, temp_max=24.0, co2_threshold=1000,
        #     humidity_min=30.0, humidity_max=60.0, effectiveness_score=1.0
        # )
        # Заглушка, если модель CompromisePreference не импортирована здесь
        class DummyCompromise:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        return DummyCompromise(
            user_count=0, temp_min=20.0, temp_max=24.0, co2_threshold=1000, # Используем 1000, как в вашем generate
            humidity_min=30.0, humidity_max=60.0, effectiveness_score=1.0
        )


class MockOccupancyAnalyzer:
    def get_next_expected_return_time(self, current_datetime): return None
    def get_expected_empty_duration(self, current_datetime): return None
    def update_patterns(self, force: bool = True): return True # Добавим метод


# --- Конфигурация Логгирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ComparativeSimulation")

# --- Параметры Симуляции ---
SIMULATION_MONTHS = 3 # Запрошенные 3 месяца
SIMULATED_DATA_CSV_PATH = "simulated_ventilation_history.csv" # Из вашего generate_simulated_data.py
MARKOV_MODEL_TRAIN_DIR = "data/markov_sim_trained" # Отдельная папка для обученной на симуляции модели
TIME_STEP_MINUTES = 2 # Должно совпадать с generate_simulated_data.py

# Энергопотребление для разных скоростей (примерные значения, Вт)
POWER_CONSUMPTION_W = {
    "off": 0,
    "low": 20,
    "medium": 50,
    "max": 100
}

# Целевые показатели комфорта
TARGET_CO2_MIN = 400
TARGET_CO2_MAX = 1000 # Более строгий порог для оценки комфорта
TARGET_TEMP_MIN = 20.0
TARGET_TEMP_MAX = 24.0

# --- Контроллеры ---

# 1. On/Off Контроллер
class OnOffController:
    def __init__(self, co2_threshold_on=950, co2_threshold_off=800, speed="max"):
        self.co2_threshold_on = co2_threshold_on
        self.co2_threshold_off = co2_threshold_off
        self.speed = speed
        self.current_action = "off"

    def decide_action(self, co2, occupants, temp): # Добавил temp для согласованности
        if occupants == 0:
            self.current_action = "off"
            return "off"
        if co2 > self.co2_threshold_on:
            self.current_action = self.speed
        elif co2 < self.co2_threshold_off:
            self.current_action = "off"
        # Иначе сохраняем текущее состояние (гистерезис)
        return self.current_action

# 2. PID Контроллер (упрощенный)
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint_co2, time_step_minutes):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint_co2 = setpoint_co2
        self.time_step_seconds = time_step_minutes * 60
        self.integral = 0
        self.previous_error = 0

    def decide_action(self, co2, occupants, temp): # Добавил temp
        if occupants == 0:
            self.integral = 0 # Сброс интеграла при отсутствии людей
            self.previous_error = 0
            return "off"

        error = self.setpoint_co2 - co2
        self.integral += error * self.time_step_seconds
        derivative = (error - self.previous_error) / self.time_step_seconds
        self.previous_error = error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Маппинг output на скорости (примерный)
        # Эти пороги нужно будет подбирать
        if output <= 0: # CO2 ниже или равен уставке
            return "off"
        elif output < 50: # Небольшое превышение
             return "low"
        elif output < 150: # Среднее превышение
             return "medium"
        else: # Сильное превышение
             return "max"

# 3. Fuzzy Logic Контроллер
class FuzzyController:
    def __init__(self):
        # Входы
        self.co2 = ctrl.Antecedent(np.arange(400, 2001, 50), 'co2') # ppm
        self.occupants = ctrl.Antecedent(np.arange(0, 6, 1), 'occupants') # количество людей

        # Выход
        self.ventilation = ctrl.Consequent(np.arange(0, 101, 1), 'ventilation_power') # % мощности

        # Функции принадлежности для CO2
        self.co2['low'] = fuzz.trimf(self.co2.universe, [400, 400, 800])
        self.co2['medium'] = fuzz.trimf(self.co2.universe, [700, 1000, 1300])
        self.co2['high'] = fuzz.trimf(self.co2.universe, [1200, 2000, 2000])

        # Функции принадлежности для Occupants
        self.occupants['zero'] = fuzz.trimf(self.occupants.universe, [0, 0, 1])
        self.occupants['few'] = fuzz.trimf(self.occupants.universe, [0, 1, 3])
        self.occupants['many'] = fuzz.trimf(self.occupants.universe, [2, 5, 5])

        # Функции принадлежности для Ventilation Power
        self.ventilation['off'] = fuzz.trimf(self.ventilation.universe, [0, 0, 10])
        self.ventilation['low'] = fuzz.trimf(self.ventilation.universe, [5, 25, 45])
        self.ventilation['medium'] = fuzz.trimf(self.ventilation.universe, [40, 60, 80])
        self.ventilation['high'] = fuzz.trimf(self.ventilation.universe, [75, 100, 100])

        # Правила
        self.rule1 = ctrl.Rule(self.occupants['zero'], self.ventilation['off'])
        self.rule2 = ctrl.Rule(self.co2['low'] & self.occupants['few'], self.ventilation['low'])
        self.rule3 = ctrl.Rule(self.co2['low'] & self.occupants['many'], self.ventilation['low'])
        self.rule4 = ctrl.Rule(self.co2['medium'] & self.occupants['few'], self.ventilation['medium'])
        self.rule5 = ctrl.Rule(self.co2['medium'] & self.occupants['many'], self.ventilation['high'])
        self.rule6 = ctrl.Rule(self.co2['high'], self.ventilation['high']) # Если CO2 высокий, всегда высокая вентиляция (если есть люди)
        self.rule7 = ctrl.Rule(self.co2['low'] & self.occupants['zero'], self.ventilation['off']) # Дублирует правило 1, но для явности

        self.ventilation_ctrl_system = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3, self.rule4, self.rule5, self.rule6, self.rule7])
        self.simulation = ctrl.ControlSystemSimulation(self.ventilation_ctrl_system)

    def decide_action(self, co2, occupants, temp): # Добавил temp
        if occupants == 0:
            return "off"

        try:
            self.simulation.input['co2'] = co2
            self.simulation.input['occupants'] = occupants
            self.simulation.compute()
            power_output = self.simulation.output['ventilation_power']
        except Exception as e:
            # logger.warning(f"Fuzzy computation error: {e}. Inputs: CO2={co2}, Occupants={occupants}. Defaulting to off.")
            # Это может происходить, если входные значения выходят за пределы universe.
            # Нужно либо расширить universe, либо "обрезать" входные значения.
            # Пока просто вернем "off"
            return "off"


        # Маппинг % мощности на скорости
        if power_output <= 10: return "off"
        elif power_output <= 40: return "low"
        elif power_output <= 70: return "medium"
        else: return "max"

# --- Обучение Markov Модели ---
def train_markov_model_on_simulation(csv_path, model_dir, epochs=1, learning_rate=0.1):
    logger.info("Starting Markov model training on simulated data...")
    os.makedirs(model_dir, exist_ok=True)

    mock_data_manager = MockDataManager()
    mock_pico_manager = MockPicoManager()
    mock_preference_manager = MockPreferenceManager()
    mock_occupancy_analyzer = MockOccupancyAnalyzer()

    markov_controller = MarkovController(
        data_manager=mock_data_manager,
        pico_manager=mock_pico_manager,
        preference_manager=mock_preference_manager,
        occupancy_analyzer=mock_occupancy_analyzer,
        model_dir=model_dir
    )
    markov_controller.learning_rate = learning_rate
    initial_model_path = markov_controller.model_file # Путь, куда контроллер сохранит свою модель
    logger.info(f"MarkovController will use/create model at: {initial_model_path}")


    if not os.path.exists(csv_path):
        logger.error(f"Simulated data CSV for training not found: {csv_path}")
        return None

    for epoch in range(epochs):
        logger.info(f"--- Markov Training Epoch {epoch + 1}/{epochs} ---")
        previous_csv_row = None
        rows_processed = 0
        # Инициализация состояния перед первой строкой
        with open(csv_path, 'r', newline='') as csvfile_init:
            reader_init = csv.DictReader(csvfile_init)
            first_row = next(reader_init, None)
            if first_row:
                mock_data_manager.update_sensor_data_from_row(first_row)
                markov_controller.current_state = markov_controller._evaluate_state() # Устанавливаем начальное состояние

        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # Пропускаем первую строку, так как она уже использована для инициализации previous_state
            # или если мы не делаем так, то previous_csv_row = first_row
            # и начинаем цикл со второй строки, чтобы иметь предыдущее и текущее.
            # Правильнее:
            # mock_data_manager.update_sensor_data_from_row(first_row)
            # previous_state_key = markov_controller._evaluate_state()

            for current_csv_row in reader:
                if previous_csv_row is None: # Для самой первой строки
                    previous_csv_row = current_csv_row
                    # Обновляем data_manager на основе previous_csv_row, чтобы _evaluate_state() дало правильный previous_state_key
                    mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                    previous_state_key = markov_controller._evaluate_state()
                    continue


                # Состояние до действия (основано на previous_csv_row)
                mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                previous_state_key = markov_controller._evaluate_state()

                # Действие, которое привело к current_csv_row (взято из previous_csv_row, так как оно действовало *до* current)
                action_taken_str = previous_csv_row.get('ventilation_action', 'off') # Действие из предыдущего шага

                # Состояние после действия (основано на current_csv_row)
                mock_data_manager.update_sensor_data_from_row(current_csv_row)
                current_state_key = markov_controller._evaluate_state()


                if previous_state_key and current_state_key and action_taken_str:
                    try:
                        action_enum = MarkovAction(action_taken_str)
                        # logger.debug(f"Training: {previous_state_key} --({action_enum.value})--> {current_state_key}")
                        markov_controller._update_model(previous_state_key, action_enum.value, current_state_key, reward=None) # Reward не используется в _update_model
                        rows_processed +=1
                    except ValueError:
                        logger.warning(f"Invalid action string '{action_taken_str}' in training data row: {previous_csv_row}")
                else:
                    logger.warning(f"Skipping training step due to missing data. Prev: {previous_state_key}, Curr: {current_state_key}, Action: {action_taken_str}")

                previous_csv_row = current_csv_row

        logger.info(f"Markov Training Epoch {epoch + 1} completed. Processed {rows_processed} transitions.")

    # Сохраняем модель после всех эпох (контроллер может сохранять и внутри _update_model)
    final_model_path = os.path.join(model_dir, "markov_model_trained_on_sim.json") # Другое имя, чтобы не перезаписать исходную
    try:
        with open(final_model_path, 'w') as f:
            json.dump(markov_controller.transition_model, f, indent=2)
        logger.info(f"Trained Markov model saved to: {final_model_path}")
    except Exception as e:
        logger.error(f"Error saving trained Markov model: {e}")

    return markov_controller # Возвращаем обученный экземпляр


# --- Симуляция Сравнения ---
def run_comparative_simulation(simulated_data_path, trained_markov_controller, time_step_minutes):
    logger.info("Starting comparative simulation...")

    if not os.path.exists(simulated_data_path):
        logger.error(f"Simulated data for comparison not found: {simulated_data_path}")
        return

    # Инициализация контроллеров
    on_off_ctrl = OnOffController(co2_threshold_on=950, co2_threshold_off=800, speed="max")
    pid_ctrl = PIDController(Kp=0.5, Ki=0.01, Kd=0.1, setpoint_co2=750, time_step_minutes=time_step_minutes) # Kp,Ki,Kd нужно подбирать!
    fuzzy_ctrl = FuzzyController()
    markov_ctrl = trained_markov_controller # Используем уже обученный

    controllers = {
        "OnOff": on_off_ctrl,
        "PID": pid_ctrl,
        "Fuzzy": fuzzy_ctrl,
        "Markov": markov_ctrl
    }

    # Сбор статистики
    stats = {name: {
        "total_energy_kwh": 0,
        "time_steps": 0,
        "time_co2_good": 0, # CO2 < TARGET_CO2_MAX
        "time_co2_medium": 0, # TARGET_CO2_MAX <= CO2 < 1200
        "time_co2_high": 0, # CO2 >= 1200
        "time_temp_good": 0, # TARGET_TEMP_MIN <= Temp <= TARGET_TEMP_MAX
        "time_temp_cold": 0, # Temp < TARGET_TEMP_MIN
        "time_temp_hot": 0,  # Temp > TARGET_TEMP_MAX
        "actions_count": {"off": 0, "low": 0, "medium": 0, "max": 0},
        "total_co2": 0,
        "total_temp": 0
    } for name in controllers}

    simulation_duration_days = (SIMULATION_MONTHS * 30) # Примерно
    num_records_to_process = (simulation_duration_days * 24 * 60) / time_step_minutes

    records_processed = 0

    with open(simulated_data_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if records_processed >= num_records_to_process:
                logger.info(f"Reached simulation target of {simulation_duration_days} days ({records_processed} records).")
                break
            try:
                timestamp = datetime.fromisoformat(row['timestamp'])
                co2 = float(row['co2'])
                temp = float(row['temperature'])
                # humidity = float(row.get('humidity', 50.0)) # Если есть влажность
                occupants = int(row['occupants'])
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping row due to parsing error: {row} - {e}")
                continue

            # Для Markov контроллера обновляем его внутренний data_manager
            if hasattr(markov_ctrl, 'data_manager') and isinstance(markov_ctrl.data_manager, MockDataManager):
                 markov_ctrl.data_manager.update_sensor_data_from_row(row)
                 # Markov контроллер также должен вызывать _evaluate_state для обновления своего внутреннего состояния
                 # и порогов перед _decide_action.
                 # _evaluate_state вызывается внутри _decide_action или перед ним в логике MarkovController
                 # Для симуляции, мы должны обеспечить, что его _evaluate_state() будет вызван с текущими данными
                 # и он НЕ будет пытаться управлять pico_manager или ждать self.scan_interval.
                 # Его метод _decide_action() должен быть достаточен, если он вызывает _evaluate_state()
                 markov_ctrl.current_state = markov_ctrl._evaluate_state() # Явный вызов для обновления порогов


            for name, controller_instance in controllers.items():
                current_stats = stats[name]
                current_stats["time_steps"] += 1

                # Получение решения от контроллера
                if name == "Markov":
                    action = controller_instance._decide_action() # Markov использует свое внутреннее состояние
                    if isinstance(action, MarkovAction): # Если _decide_action вернул Enum
                        action = action.value
                else:
                    action = controller_instance.decide_action(co2, occupants, temp)

                if action not in POWER_CONSUMPTION_W:
                    logger.warning(f"Controller {name} returned invalid action: {action}. Defaulting to 'off'.")
                    action = "off"

                current_stats["actions_count"][action] +=1

                # Энергопотребление
                power_w = POWER_CONSUMPTION_W[action]
                energy_wh = power_w * (time_step_minutes / 60.0)
                current_stats["total_energy_kwh"] += energy_wh / 1000.0

                # Качество воздуха (CO2)
                current_stats["total_co2"] += co2
                if co2 < TARGET_CO2_MAX:
                    current_stats["time_co2_good"] += 1
                elif co2 < 1200: # Порог для "среднего"
                    current_stats["time_co2_medium"] += 1
                else:
                    current_stats["time_co2_high"] += 1

                # Температурный комфорт
                current_stats["total_temp"] += temp
                if TARGET_TEMP_MIN <= temp <= TARGET_TEMP_MAX:
                    current_stats["time_temp_good"] += 1
                elif temp < TARGET_TEMP_MIN:
                    current_stats["time_temp_cold"] += 1
                else:
                    current_stats["time_temp_hot"] += 1
            records_processed +=1
            if records_processed % (1440 / time_step_minutes * 7) == 0: # Лог каждую неделю симуляции
                 logger.info(f"Simulated {records_processed * time_step_minutes / (60*24):.1f} days...")


    # --- Вывод Результатов ---
    logger.info("\n--- Comparative Simulation Results ---")
    for name, result_stats in stats.items():
        logger.info(f"\nController: {name}")
        logger.info(f"  Total Energy Consumed: {result_stats['total_energy_kwh']:.2f} kWh")
        total_steps = result_stats['time_steps']
        if total_steps > 0:
            logger.info(f"  Average CO2: {result_stats['total_co2'] / total_steps:.0f} ppm")
            logger.info(f"  Average Temperature: {result_stats['total_temp'] / total_steps:.1f} °C")
            logger.info(f"  Time CO2 Good (<{TARGET_CO2_MAX} ppm): {(result_stats['time_co2_good']/total_steps)*100:.1f}%")
            logger.info(f"  Time CO2 Medium ({TARGET_CO2_MAX}-1200 ppm): {(result_stats['time_co2_medium']/total_steps)*100:.1f}%")
            logger.info(f"  Time CO2 High (>=1200 ppm): {(result_stats['time_co2_high']/total_steps)*100:.1f}%")
            logger.info(f"  Time Temp Good ({TARGET_TEMP_MIN}-{TARGET_TEMP_MAX}°C): {(result_stats['time_temp_good']/total_steps)*100:.1f}%")
            logger.info(f"  Time Temp Cold (<{TARGET_TEMP_MIN}°C): {(result_stats['time_temp_cold']/total_steps)*100:.1f}%")
            logger.info(f"  Time Temp Hot (>{TARGET_TEMP_MAX}°C): {(result_stats['time_temp_hot']/total_steps)*100:.1f}%")
            logger.info(f"  Action Distribution: {result_stats['actions_count']}")
        else:
            logger.info("  No simulation steps processed for this controller.")

def main_simulation_script():
    # 1. Убедиться, что данные сгенерированы
    if not os.path.exists(SIMULATED_DATA_CSV_PATH):
        logger.info(f"'{SIMULATED_DATA_CSV_PATH}' not found. Please run 'generate_simulated_data.py' first.")
        # Здесь можно добавить вызов generate_simulated_data.main() если он есть
        # from generate_simulated_data import main as generate_main
        # generate_main()
        # if not os.path.exists(SIMULATED_DATA_CSV_PATH):
        #     logger.error("Failed to generate simulated data. Exiting.")
        #     return
        return # Пока просто выйдем, если файла нет

    # 2. Прогрев/Обучение Markov Модели
    # Обучаем на всем датасете, который будет использоваться для симуляции,
    # это даст ему "представление" о паттернах в этих данных.
    # Используем 2 эпохи для лучшего обучения.
    trained_markov_ctrl = train_markov_model_on_simulation(
        SIMULATED_DATA_CSV_PATH,
        MARKOV_MODEL_TRAIN_DIR,
        epochs=2, # Количество эпох обучения на данных
        learning_rate=0.15 # Чуть выше, чем в оригинальном train_markov_model
    )

    if trained_markov_ctrl is None:
        logger.error("Failed to train Markov model. Exiting.")
        return

    # 3. Запуск симуляции сравнения
    run_comparative_simulation(SIMULATED_DATA_CSV_PATH, trained_markov_ctrl, TIME_STEP_MINUTES)

if __name__ == "__main__":
    main_simulation_script()