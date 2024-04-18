#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class Reader_Bitocin_data_csv {
private:
    std::string filename;
    char delimiter;

public:
    Reader_Bitocin_data_csv(const std::string& filename, char delimiter = ',') : filename(filename), delimiter(delimiter) {}

    std::vector<std::vector<std::string>> getData() {
        std::vector<std::vector<std::string>> data;

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return data;
        }

        // Ignore the first line
        std::string line;
        std::getline(file, line);

        // Read the rest of the lines
        while (std::getline(file, line)) {
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string cell;

            // Read the first column as string
            std::getline(ss, cell, delimiter);
            row.push_back(cell);

            // Read the rest of the columns as floats
            while (std::getline(ss, cell, delimiter)) {
                try {
                    row.push_back(cell);
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Warning: Invalid float value in line: " << line << std::endl;
                }
            }

            if (row.size() == 7) {
                data.push_back(row);
            } else {
                std::cerr << "Warning: Invalid number of columns in line: " << line << std::endl;
            }
        }

        file.close();
        return data;
    }
};

int main() {
    Reader_Bitocin_data_csv reader("../BTC_USD_1D/BTCUSD_HISTORICAL_1D_2015-01-01_2015-12-31.csv");
    std::vector<std::vector<std::string>> data = reader.getData();

    // Print the data
    for (const auto& row : data) {
        for (const auto& cell : row) {
            std::cout << cell << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}
