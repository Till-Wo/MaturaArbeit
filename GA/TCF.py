
import xlsxwriter, time


def print_value(name, value):
    space = ""
    for _ in range(19 - len(name)):
        space += " "
    print(f"{name}:{space}{value}")


class Writer:
    def __init__(self, name, worksheetname=None):
        self.row_names = {}
        self.column_names = {}
        self.workbook = xlsxwriter.Workbook(f"Data\\{name}.xlsx")
        if worksheetname is not None:
            self.worksheet = self.workbook.add_worksheet(worksheetname)
        else:
            self.worksheet = self.workbook.add_worksheet()


    def save(self, row_key: str, column_key: str, data):
        if not row_key in self.row_names:
            self.row_names[row_key] = len(self.row_names)+1
            self.worksheet.write(self.row_names[row_key], 0, row_key)
        if not column_key in self.column_names:
            self.column_names[column_key] = len(self.column_names)+1
            self.worksheet.write(0, self.column_names[column_key], column_key)

        self.worksheet.write(self.row_names[row_key], self.column_names[column_key], data)
    def close(self):
        self.workbook.close()



class TimeIt:
    def __init__(self):
        self.start_time = time.time()
        self.stared_at = time.time()
        self.average = 0

    def reset(self):
        self.stared_at = time.time()


    def time_since_start(self):
        return time.time()-self.start_time

    def update(self):
        return time.time()-self.stared_at

    def update_and_reset(self):
        t = time.time()-self.stared_at
        self.reset()
        return t

    def print(self, seconds, name="Time"):
        minutes, hours, days = 0, 0, 0
        if seconds >= 60:
            minutes = seconds // 60
            seconds = seconds % 60
            if minutes >= 60:
                hours = minutes // 60
                minutes = minutes % 60
                if hours >= 24:
                    days = hours // 24
                    hours = hours % 24

        seconds = round(seconds, 2)
        print_value(name, f"{int(days)}d: {int(hours)}h: {int(minutes)}min: {seconds}s:")

    def print_Duration(self, reset=False):
        seconds = self.update()
        self.print(seconds, "Duration")
        if reset:
            self.reset()












































