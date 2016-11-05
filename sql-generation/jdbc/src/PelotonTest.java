//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// PelotonTest.java
//
// Identification: script/testing/nops/src/PelotonTest.java
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

import java.sql.*;
import java.util.Arrays;
import java.util.Random;
import java.util.Date;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.*;
import java.util.List;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.HashMap;
import java.util.ArrayList;

public class PelotonTest {

	// Peloton, Postgres, Timesten Endpoints
	private final String[] url = {
			"jdbc:postgresql://localhost:5432/postgres", // PELOTON
			"jdbc:postgresql://localhost:5432/postgres", // POSTGRES
			"jdbc:timesten:client:TTC_SERVER_DSN=xxx;TTC_SERVER=xxx;TCP_PORT=xxx" }; // TIMESTEN

	private final String[] user = { "postgres", "postgres", "xxx" };
	private final String[] pass = { "postgres", "postgres", "xxx" };
	private final String[] driver = { "org.postgresql.Driver",
			"org.postgresql.Driver", "com.timesten.jdbc.TimesTenDriver" };

	private final int LOG_LEVEL = 0;
	
	private Map<String, List<String> > dict = new HashMap<String, List<String> >();
	private Map<String, Map<String, Integer> > min_dict = new HashMap<String, Map<String, Integer> >();
	private Map<String, Map<String, Integer> > max_dict = new HashMap<String, Map<String, Integer> >();

	// Query types
	public static final int SINGLE_COLUMN = 0;
	public static final int TWO_COLUMNS = 1;

	public static int numThreads = 1;

	// Endpoint types
	public static final int PELOTON = 0;
	public static final int POSTGRES = 1;
	public static final int TIMESTEN = 2;

	// QUERY TEMPLATES
	private final String DROP = "DROP TABLE IF EXISTS A;";

	private final String DDL = "CREATE TABLE A (id INT PRIMARY KEY, data VARCHAR(100), "
			+ "field1 VARCHAR(100), field2 VARCHAR(100), field3 VARCHAR(100), field4 VARCHAR(100), "
			+ "field5 VARCHAR(100), field6 VARCHAR(100), field7 VARCHAR(100), field8 VARCHAR(100), field9 VARCHAR(100));";

	private final String INDEXSCAN_PARAM = "SELECT * FROM A WHERE id = ?";

	private final String UPDATE_BY_INDEXSCAN = "UPDATE A SET id=99 WHERE id=?";

	private final String UPDATE_BY_LARGE_DATA = "UPDATE A SET data = ?, "
			+ "field1 = ?, field2 = ?, field3 = ?, field4 = ?, "
			+ "field5 = ?, field6 = ?, field7 = ?, field8 = ?, field9 = ? WHERE id = 99;";

	private final String LARGE_STRING = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
			+ "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

	private final Connection conn;

	private enum TABLE {
		A, B, AB
	}

	private int target;
	private int query_type;

	class QueryWorker extends Thread {
		public long runningTime;
		public long totalOps = 0;
		public Connection connection;

		public QueryWorker(long runningTime) {
			this.runningTime = runningTime;
			try {
				this.connection = DriverManager.getConnection(url[target],
						user[target], pass[target]);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		class RandomColumnPredicate {
			public String table_;
			public String column_;
			public int low_;
			public int high_;
			
			public RandomColumnPredicate(String table, String column, int low, int high) {
				table_ = table;
				column_ = column;
				low_ = low;
				high_ = high;
			}
		}
		
		private RandomColumnPredicate GetRandomColumnPredicate(Map<String, List<String> > dict,
				Map<String, Map<String, Integer> > min_dict,
				Map<String, Map<String, Integer> > max_dict,
				Random random, String key) {
			String randomKey = null;
			if (key != null) {
				randomKey = key;
			} else {
				List<String> keys = new ArrayList<String>(dict.keySet());
				randomKey = keys.get( random.nextInt(keys.size()) );
			}

			List<String> columns = dict.get(randomKey);
			String randomColumn = columns.get( random.nextInt(columns.size()) );
			int min = min_dict.get(randomKey).get(randomColumn).intValue();
			int max = max_dict.get(randomKey).get(randomColumn).intValue();
			
			int v1 = random.nextInt((max - min) + 1) + min;
			int v2 = random.nextInt((max - min) + 1) + min;
			if (v1 > v2) {
				int tmp = v1;
				v1 = v2;
				v2 = tmp;
			}
			
			return new RandomColumnPredicate(randomKey, randomColumn, v1, v2);
			
		}

		public void run() {
			try {
				Statement stmt = connection.createStatement();
				int num_ops = 300;

				Random random = new Random();
				//random.setSeed(new Date().getTime());
				
				for (int i = 0; i < num_ops; i++) {
					RandomColumnPredicate predicate = 
							GetRandomColumnPredicate(dict, min_dict, max_dict, random, null);
					
					String query = "select count(*) from " + predicate.table_ +
							" where " + predicate.column_ + ">" + predicate.low_ +
							" and " + predicate.column_ + "<" + predicate.high_;
					ResultSet rs = stmt.executeQuery(query);
					
					
					while (rs.next()) {
						System.out.println(rs.getInt(1));
					}
					
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public PelotonTest(int target, int query_type) throws SQLException {
		this.target = target;
		this.query_type = query_type;
		try {
			Class.forName(driver[target]);
			if (LOG_LEVEL != 0) {
				org.postgresql.Driver.setLogLevel(LOG_LEVEL);
				DriverManager.setLogStream(System.out);
			}
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		conn = this.makeConnection();
		return;
	}

	private Connection makeConnection() throws SQLException {
		Connection conn = DriverManager.getConnection(url[target],
				user[target], pass[target]);
		return conn;
	}

	public void Close() throws SQLException {
		conn.close();
	}

	/**
	 * Collect ranges of each column
	 *
	 * @throws SQLException
	 */
	public void Init() throws SQLException, FileNotFoundException {
		
	    //Scanner Example - read file line by line in Java using Scanner
        FileInputStream fis = new FileInputStream("../ddl.txt");
        Scanner scanner = new Scanner(fis);
      
        //reading file line by line using Scanner in Java
        System.out.println("Reading file line by line in Java using Scanner");
        
        String table = null;
        while(scanner.hasNextLine()){
            String token = scanner.nextLine();
            if (token.equals("New Table:")) {
            	table = null;
            	continue;
            }
            
            if (table == null) {
            	table = token;
            	dict.put(table, new ArrayList<String>());
            	continue;
            }

            dict.get(table).add(token);
        }
      
        scanner.close();
        

        // Getting min and max values
		conn.setAutoCommit(true);
		Statement stmt = conn.createStatement();

        for (String key: dict.keySet()) {
        	System.out.println(key);
            min_dict.put(key, new HashMap<String, Integer>());
            max_dict.put(key, new HashMap<String, Integer>());
        	for (String column: dict.get(key)) {
        		System.out.println(column);

        	    // get min
        		ResultSet rs = stmt.executeQuery("select min(" + column + ") from " + key);
        	    while (rs.next()) {
        	    	System.out.println(rs.getInt(1));
        	    	min_dict.get(key).put(column, rs.getInt(1));
        	    }

        	    // get max
        		rs = stmt.executeQuery("select max(" + column + ") from " + key);
        	    while (rs.next()) {
        	    	System.out.println(rs.getInt(1));
        	    	max_dict.get(key).put(column, rs.getInt(1));
        	    }
        	}
        }
	}

	public void Timed_Nop_Test() throws Exception {
		int runningTime = 15;
		QueryWorker[] workers = new QueryWorker[numThreads];
		// Initialize all worker threads
		for (int i = 0; i < numThreads; i++) {
			workers[i] = new QueryWorker(1000 * runningTime);
		}
		// Submit to thread pool
		ExecutorService executorPool = Executors.newFixedThreadPool(numThreads);
		for (int i = 0; i < numThreads; i++) {
			executorPool.submit(workers[i]);
		}
		// No more task to submit
		executorPool.shutdown();
		// Wait for tasks to terminate
		executorPool.awaitTermination(runningTime + 3, TimeUnit.SECONDS);
		// Calculate the total number of ops
		long totalThroughput = 0;
		for (int i = 0; i < numThreads; i++) {
			totalThroughput += workers[i].totalOps * 1.0 / runningTime;
		}
		System.out.println(totalThroughput);
	}

	private void PerformNopQuery(Statement stmt, long numOps) throws Exception {
		for (long i = 0; i < numOps; i++) {
			try {
				stmt.execute(";");
			} catch (Exception e) {
			}
		}
	}

	static private void printHelpMessage() {
		System.out
				.println("Please specify target: [peloton|timesten|postgres] "
						+ "[semicolon|select|batch_update|simple_update|large_update]");
	}

	static public void main(String[] args) throws Exception {
		if (args.length < 2) {
			printHelpMessage();
			return;
		}

		int target = PELOTON;
		switch (args[0]) {
		case ("peloton"): {
			target = PELOTON;
			break;
		}
		default: {
			printHelpMessage();
			return;
		}
		}

		int query_type = SINGLE_COLUMN;
		switch (args[1]) {
		case "single_column": {
			query_type = SINGLE_COLUMN;
			break;
		}
		case "two_columns": {
			query_type = TWO_COLUMNS;
			break;
		}
		default: {
			printHelpMessage();
			return;
		}
		}

		PelotonTest pt = new PelotonTest(target, query_type);
		pt.Init();

		numThreads = Integer.parseInt(args[2]);
		pt.Close();
		pt.Timed_Nop_Test();
	}

}
