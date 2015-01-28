from sqlalchemy import *


class DB():


    def __init__(self, dbname):
        self.db = create_engine('sqlite:///'+str(dbname)+'.db')
        self.db.echo = False
        self.metadata = BoundMetaData(db)
        self.node_weights = Table('node_weights', self.metadata, autoload=True)
        self.node_averages = Table('node_averages', self.metadata, autload=True)
        self.session = sessionmaker(bind=db)

    def create_structure(self):
        self.node_weights = Table("node_weights", self.metadata,
                             Column("id", Integer, primary_key=True, autoincrement=True),
                             Column("graph_name", String),
                             Column("a", Float),
                             Column("b", Float),
                             Column("weight", Float),
                             Column("run", Integer),
                             Column("nid", Integer),
                             Column("percentage", Float),
                             Column("mult_factor", Float)
        )

        self.node_averages = Table("node_averages", self.metadata,
                                   Column("id", Integer, primary_key=True, autoincrement=True),
                                   Column("graph_name", String),
                                   Column("a", Float),
                                   Column("b", Float),
                                   Column("average_weight", Float),
                                   Column("num_runs", Integer),
                                   Column("nid", Integer),
                                   Column("percentage", Float),
                                   Column("mult_factor", Float)
        )


    def load_db(self):
        self.node_weights = Table('node_weights', self.metadata, autoload=True)
        self.node_averages = Table('node_averages', self.metadata, autload=True)


    def write_weights(self, graph_name, a, b, weights, run, percentage, mult_factor):
        for idx, w in enumerate(weights):
            self.session.add(self.node_weights(graph_name=graph_name, a=a, b=b, weight=w, run=run, nid=idx, percentage=percentage, mult_factor=mult_factor))
        self.session.commit()
