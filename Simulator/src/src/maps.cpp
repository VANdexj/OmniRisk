#include "maps.hpp"

using namespace mocka;

void
Maps::randomMapGenerate()
{

  std::default_random_engine eng(info.seed);

  double _resolution = 1 / info.scale;

  double _x_l = -info.sizeX / (2 * info.scale);
  double _x_h = info.sizeX / (2 * info.scale);
  double _y_l = -info.sizeY / (2 * info.scale);
  double _y_h = info.sizeY / (2 * info.scale);
  double _h_l = 0;
  double _h_h = info.sizeZ / info.scale;

  std::uniform_real_distribution<double> rand_x;
  std::uniform_real_distribution<double> rand_y;
  std::uniform_real_distribution<double> rand_w;
  std::uniform_real_distribution<double> rand_h;

  pcl::PointXYZ pt_random;

  rand_x = std::uniform_real_distribution<double>(_x_l, _x_h);
  rand_y = std::uniform_real_distribution<double>(_y_l, _y_h);
  rand_w = std::uniform_real_distribution<double>(_w_l, _w_h);
  rand_h = std::uniform_real_distribution<double>(_h_l, _h_h);

  for (int i = 0; i < _ObsNum; i++)
  {
    double x, y;
    x = rand_x(eng);
    y = rand_y(eng);

    double w, h;
    w = rand_w(eng);
    h = rand_h(eng);

    int widNum = ceil(w / _resolution);
    int heiNum = ceil(h / _resolution);

    int rl, rh, sl, sh;
    rl = -widNum / 2;
    rh = widNum / 2;
    sl = -widNum / 2;
    sh = widNum / 2;

    for (int r = rl; r < rh; r++)
      for (int s = sl; s < sh; s++)
      {
        for (int t = 0; t < heiNum; t++)
        {
          if ((r - rl) * (r - rh + 1) * (s - sl) * (s - sh + 1) * t *
                (t - heiNum + 1) ==
              0)
          {
            pt_random.x = x + r * _resolution;
            pt_random.y = y + s * _resolution;
            pt_random.z = t * _resolution;
            info.cloud->points.push_back(pt_random);
          }
        }
      }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud = generateGround(info.cloud, _resolution);
  *info.cloud += *ground_cloud;
  
  info.cloud->width    = info.cloud->points.size();
  info.cloud->height   = 1;
  info.cloud->is_dense = true;
}

void
Maps::wall()
{
  std::default_random_engine eng(info.seed);

  double _resolution = 1 / info.scale;

  double _x_l = -info.sizeX / (2 * info.scale);
  double _x_h = info.sizeX / (2 * info.scale);
  double _y_l = -info.sizeY / (2 * info.scale);
  double _y_h = info.sizeY / (2 * info.scale);
  double _h_l = 0.3 * info.sizeZ / info.scale;
  double _h_h = info.sizeZ / info.scale;

  std::uniform_real_distribution<double> rand_x(_x_l, _x_h);
  std::uniform_real_distribution<double> rand_y(_y_l, _y_h);
  std::uniform_real_distribution<double> rand_w(_wall_w_l, _wall_w_h);
  std::uniform_real_distribution<double> rand_h(_h_l, _h_h);
  std::uniform_real_distribution<float> rand_yaw(-M_PI, M_PI);
  std::uniform_real_distribution<float> rand_pitch(-M_PI / 10, M_PI / 10);

  pcl::PointXYZ pt_random;
  for (int i = 0; i < _wall_num; ++i)
  {
    float cx = rand_x(eng);
    float cy = rand_y(eng);
    float cz = 0.0f;
    float yaw = rand_yaw(eng);
    float pitch = rand_pitch(eng);
    float width = rand_w(eng);
    float height = rand_h(eng);
    float thick = _wall_thick;

    int w_steps = std::ceil(width / _resolution);
    int h_steps = std::ceil(height / _resolution);
    int t_steps = std::ceil(thick / _resolution);

    float w0 = -width / 2.0f;
    float h0 = 0.0f;
    float t0 = -thick / 2.0f;

    float cosy = std::cos(yaw), siny = std::sin(yaw);
    float cosp = std::cos(pitch), sinp = std::sin(pitch);

    for (int i = 0; i < w_steps; ++i)
    {
      for (int j = 0; j < h_steps; ++j)
      {
        for (int k = 0; k < t_steps; ++k)
        {
          float x_local = w0 + i * _resolution;
          float y_local = t0 + k * _resolution;
          float z_local = h0 + j * _resolution;

          float x1 = cosy * x_local - siny * y_local;
          float y1 = siny * x_local + cosy * y_local;
          float z1 = z_local;

          float y2 = cosp * y1 - sinp * z1;
          float z2 = sinp * y1 + cosp * z1;
          float x2 = x1;

          pcl::PointXYZ pt;
          pt.x = cx + x2;
          pt.y = cy + y2;
          pt.z = cz + z2;

          info.cloud->points.push_back(pt);
        }
      }
    }
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud = generateGround(info.cloud, _resolution);
  *info.cloud += *ground_cloud;

  if (_wall_ceiling){
    pcl::PointCloud<pcl::PointXYZ>::Ptr ceiling_cloud = generateGround(info.cloud, _resolution, _h_h);
    *info.cloud += *ceiling_cloud;
  }
  
  info.cloud->width    = info.cloud->points.size();
  info.cloud->height   = 1;
  info.cloud->is_dense = true;
}

void
Maps::perlin3D()
{
  info.cloud->width  = info.sizeX * info.sizeY * info.sizeZ;
  info.cloud->height = 1;
  info.cloud->points.resize(info.cloud->width * info.cloud->height);
  double _resolution = 1 / info.scale;

  PerlinNoise noise(info.seed);

  std::vector<double>* v = new std::vector<double>;
  v->reserve(info.cloud->width);
  for (int i = 0; i < info.sizeX; ++i)
  {
    for (int j = 0; j < info.sizeY; ++j)
    {
      for (int k = 0; k < info.sizeZ; ++k)
      {
        double tnoise = 0;
        for (int it = 1; it <= fractal; ++it)
        {
          int    dfv = pow(2, it);
          double ta  = attenuation / it;
          tnoise += ta * noise.noise(dfv * i * complexity,
                                     dfv * j * complexity,
                                     dfv * k * complexity);
        }
        v->push_back(tnoise);
      }
    }
  }
  std::sort(v->begin(), v->end());
  int    tpos = info.cloud->width * (1 - fill);
  double tmp  = v->at(tpos);
  // printf("threshold: %lf", tmp);

  int pos = 0;
  for (int i = 0; i < info.sizeX; ++i)
  {
    for (int j = 0; j < info.sizeY; ++j)
    {
      for (int k = 0; k < info.sizeZ; ++k)
      {
        double tnoise = 0;
        for (int it = 1; it <= fractal; ++it)
        {
          int    dfv = pow(2, it);
          double ta  = attenuation / it;
          tnoise += ta * noise.noise(dfv * i * complexity,
                                     dfv * j * complexity,
                                     dfv * k * complexity);
        }
        if (tnoise > tmp)
        {
          info.cloud->points[pos].x =
            i / info.scale - info.sizeX / (2 * info.scale);
          info.cloud->points[pos].y =
            j / info.scale - info.sizeY / (2 * info.scale);
          info.cloud->points[pos].z = k / info.scale;
          pos++;
        }
      }
    }
  }
  info.cloud->width = pos;
  // printf("the number of points before optimization is %d", info.cloud->width);
  info.cloud->points.resize(info.cloud->width * info.cloud->height);

  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud = generateGround(info.cloud, _resolution);
  *info.cloud += *ground_cloud;
  info.cloud->width = info.cloud->points.size();
}

void
Maps::recursiveDivision(int xl, int xh, int yl, int yh, Eigen::MatrixXi& maze)
{
  // printf(
  //   "generating maze with width %d , height %d", xh - xl + 1, yh - yl + 1);

  if (xl < xh - 3 && yl < yh - 3)
  { // the remaining area is larger than or equal to 5*5, need to add both x
    // wall and y wall
    bool valid = false; // used to judge whether the wall selection is valid
    int  xm    = 0;
    int  ym    = 0;
    // printf("entered 5*5 mode");
    while (valid == false)
    {
      xm = (std::rand() % (xh - xl - 1) + xl +
            1); // generating random number between xl+1 and xh-1(pointless to
                // add a wall at the sides)
      ym = (std::rand() % (yh - yl - 1) + yl +
            1); // generating random number between yl+1 and yh-1(pointless to
                // add a wall at the sides)
      if (xl - 1 >= 0)
      { // there is a point at xl-1,ym
        if (maze(xl - 1, ym) == 0)
        { // this is an opening,need to change random number
          continue;
        }
      }

      else if (xh + 1 <= maze.cols() - 1)
      { // there is a point at xh+1,ym
        if (maze(xh + 1, ym) == 0)
        { // this is an opening,need to change random number
          continue;
        }
      }

      else if (yl - 1 >= 0)
      { // there is a point at xm,yl-1
        if (maze(xm, yl - 1) == 0)
        { // this is an opening,need to change random number
          continue;
        }
      }

      else if (yh + 1 <= maze.rows() - 1)
      { // there is a point at xm,yh+1
        if (maze(xm, yh + 1) == 0)
        { // this is an opening,need to change random number
          continue;
        }
      }

      valid = true;

    } // xm and ym are now the valid coordinate of the center of the wall
    for (int i = xl; i <= xh; i++)
    {
      maze(i, ym) = 1;
    }
    for (int j = yl; j <= yh; j++)
    {
      maze(xm, j) = 1;
    } // adding walls around the center point
    int d1 = std::rand() % (xm - xl) + xl;
    int d2 = std::rand() % (xh - xm) + xm + 1;
    int d3 = std::rand() % (ym - yl) + yl;
    int d4 =
      std::rand() % (yh - ym) + ym + 1; // generating four possible door points

    int decision = std::rand() % 4; // random selection of three doors
    switch (decision)
    {
      case 0:
        maze(d1, ym) = 0;
        maze(d2, ym) = 0;
        maze(xm, d3) = 0;
        break;

      case 1:
        maze(d1, ym) = 0;
        maze(d2, ym) = 0;
        maze(xm, d4) = 0;
        break;

      case 2:
        maze(d2, ym) = 0;
        maze(xm, d3) = 0;
        maze(xm, d4) = 0;
        break;

      case 3:
        maze(d1, ym) = 0;
        maze(xm, d3) = 0;
        maze(xm, d4) = 0;
        break;
    } // the doors are opened for this cell
    if (yl - 1 >= 0)
    {
      if (maze(xm, yl - 1) == 0)
      {
        maze(xm, yl) = 0;
      }
    }

    if (yh + 1 <= maze.rows() - 1)
    {
      if (maze(xm, yh + 1) == 0)
      {
        maze(xm, yh) = 0;
      }
    }

    if (xl - 1 >= 0)
    {
      if (maze(xl - 1, ym) == 0)
      {
        maze(xl, ym) = 0;
      }
    }

    if (xh + 1 <= maze.cols() - 1)
    {
      if (maze(xh + 1, ym) == 0)
      {
        maze(xh, ym) = 0;
      }
    }

    // std::cout << maze << std::endl;
    recursiveDivision(xl, xm - 1, yl, ym - 1, maze);
    recursiveDivision(xm + 1, xh, yl, ym - 1, maze);
    recursiveDivision(xl, xm - 1, ym + 1, yh, maze);
    recursiveDivision(xm + 1, xh, ym + 1, yh, maze);

    // printf("finished generating maze with width %d , height %d",
    //          xh - xl + 1,
    //          yh - yl + 1);
    // std::cout << maze << std::endl;
    return;
  } // when the remaining area is larger than or equal to 5*5

  else if (xl < xh - 2 && yl < yh - 2)
  {
    bool valid     = false; // used to judge whether the wall selection is valid
    int  xm        = 0;
    int  ym        = 0;
    int  doorcount = 0;
    xm             = (std::rand() % (xh - xl - 1) + xl +
          1); // generating random number between xl+1 and xh-1(pointless to
                          // add a wall at the sides)
    ym =
      (std::rand() % (yh - yl - 1) + yl +
       1); // generating random number between yl+1 and yh-1(pointless to
           // add a wall at the sides)
           // xm and ym are now the valid coordinate of the center of the wall
    for (int i = xl; i <= xh; i++)
    {
      maze(i, ym) = 1;
    }
    for (int j = yl; j <= yh; j++)
    {
      maze(xm, j) = 1;
    } // adding walls around the center point
    if (yl - 1 >= 0)
    {
      if (maze(xm, yl - 1) == 0)
      {
        maze(xm, yl) = 0;
        doorcount++;
      }
    }

    if (yh + 1 <= maze.rows() - 1)
    {
      if (maze(xm, yh + 1) == 0)
      {
        maze(xm, yh) = 0;
        doorcount++;
      }
    }

    if (xl - 1 >= 0)
    {
      if (maze(xl - 1, ym) == 0)
      {
        maze(xl, ym) = 0;
        doorcount++;
      }
    }

    if (xh + 1 <= maze.cols() - 1)
    {
      if (maze(xh + 1, ym) == 0)
      {
        maze(xh, ym) = 0;
        doorcount++;
      }
    }

    int d1 = std::rand() % (xm - xl) + xl;
    int d2 = std::rand() % (xh - xm) + xm + 1;
    int d3 = std::rand() % (ym - yl) + yl;
    int d4 =
      std::rand() % (yh - ym) + ym + 1; // generating four possible door points

    int decision = std::rand() % 4; // random selection of three doors
    switch (decision)
    {
      case 0:
        maze(d1, ym) = 0;
        maze(d2, ym) = 0;
        maze(xm, d3) = 0;
        break;

      case 1:
        maze(d1, ym) = 0;
        maze(d2, ym) = 0;
        maze(xm, d4) = 0;
        break;

      case 2:
        maze(d2, ym) = 0;
        maze(xm, d3) = 0;
        maze(xm, d4) = 0;
        break;

      case 3:
        maze(d1, ym) = 0;
        maze(xm, d3) = 0;
        maze(xm, d4) = 0;
        break;
    } // the doors are opened for this cell
    // std::cout << maze << std::endl;

    // printf("finished generating maze with width %d , height %d",
            //  xh - xl + 1,
            //  yh - yl + 1);
    // std::cout << maze << std::endl;
    return;
  }

  else if (xl < xh - 1 && yl < yh - 2)
  { // the case of 3*4+
    // printf("entered 3*4+ mode");
    int doorcount = 0;
    int ym        = 0;
    for (int i = yl; i <= yh; i++)
    {
      maze(xl + 1, i) = 1;
    } // filling a center wall
    if (yl - 1 >= 0)
    {
      if (maze(xl + 1, yl - 1) == 0)
      {
        maze(xl + 1, yl) = 0;
        doorcount++;
      }
    }
    if (yh + 1 <= maze.rows() - 1)
    {
      if (maze(xl + 1, yh + 1) == 0)
      {
        maze(xl + 1, yh) = 0;
        doorcount++;
      }
    } // opening doors if the wall blocks the old doors
    if (doorcount == 0)
    {
      ym               = std::rand() % (yh - yl + 1) + yl;
      maze(xl + 1, ym) = 0;
    }
  } // the case of 4+*3
  //
  else if (xl < xh - 2 && yl < yh - 1)
  { // the case of 4+*3
    // printf("entered 4+*3 mode");
    int doorcount = 0;
    int xm        = 0;
    for (int i = xl; i <= xh; i++)
    {
      maze(i, yl + 1) = 1;
    } // filling a center wall
    if (xl - 1 >= 0)
    {
      if (maze(xl - 1, yl + 1) == 0)
      {
        maze(xl, yl + 1) = 0;
        doorcount++;
      }
    }
    if (xh + 1 <= maze.cols() - 1)
    {
      if (maze(xh + 1, yl + 1) == 0)
      {
        maze(xh, yl + 1) = 0;
        doorcount++;
      }
    } // opening doors if the wall blocks the old doors
    if (doorcount == 0)
    {
      xm               = std::rand() % (xh - xl + 1) + xl;
      maze(xm, yl + 1) = 0;
    }
  } // the case of 4+*3

  else if (xl < xh - 1 && yl < yh - 1)
  { // the case of 3*3
    maze(xl + 1, yl + 1) = 1;
    return;
  }
  else
  {
    // printf("finished generating maze with width %d , height %d",
    //          xh - xl + 1,
    //          yh - yl + 1);
    return;
  }
}

void
Maps::recursizeDivisionMaze(Eigen::MatrixXi& maze)
{
  //! @todo all bugs here...
  int sx = maze.rows();
  int sy = maze.cols();

  int px, py;

  if (sx > 5)
    px = (std::rand() % (sx - 3) + 1);
  else
    return;

  if (sy > 5)
    py = (std::rand() % (sy - 3) + 1);
  else
    return;

  // printf("debug %d %d %d %d", sx, sy, px, py);

  int x1, x2, y1, y2;

  if (px != 1)
    x1 = (std::rand() % (px - 1) + 1);
  else
    x1 = 1;

  if ((sx - px - 3) > 0)
    x2 = (std::rand() % (sx - px - 3) + px + 1);
  else
    x2 = px + 1;

  if (py != 1)
    y1 = (std::rand() % (py - 1) + 1);
  else
    y1 = 1;

  if ((sy - py - 3) > 0)
    y2 = (std::rand() % (sy - py - 3) + py + 1);
  else
    y2 = py + 1;
  // printf("%d %d %d %d", x1, x2, y1, y2);

  if (px != 1 && px != (sx - 2))
  {
    for (int i = 1; i < (sy - 1); ++i)
    {
      if (i != y1 && i != y2)
        maze(px, i) = 1;
    }
  }
  if (py != 1 && py != (sy - 2))
  {
    for (int i = 1; i < (sx - 1); ++i)
    {
      if (i != x1 && i != x2)
        maze(i, py) = 1;
    }
  }
  switch (std::rand() % 4)
  {
    case 0:
      maze(x1, py) = 1;
      break;
    case 1:
      maze(x2, py) = 1;
      break;
    case 2:
      maze(px, y1) = 1;
      break;
    case 3:
      maze(px, y2) = 1;
      break;
  }

  if (px > 2 && py > 2)
  {
    Eigen::MatrixXi sub = maze.block(0, 0, px + 1, py + 1);
    recursizeDivisionMaze(sub);
    maze.block(0, 0, px, py) = sub;
  }
  if (px > 2 && (sy - py - 1) > 2)
  {
    Eigen::MatrixXi sub = maze.block(0, py, px + 1, sy - py);
    recursizeDivisionMaze(sub);
    maze.block(0, py, px + 1, sy - py) = sub;
  }
  if (py > 2 && (sx - px - 1) > 2)
  {
    Eigen::MatrixXi sub = maze.block(px, 0, sx - px, py + 1);
    recursizeDivisionMaze(sub);
    maze.block(px, 0, sx - px, py + 1) = sub;
  }
  if ((sx - px - 1) > 2 && (sy - py - 1) > 2)
  {

    Eigen::MatrixXi sub = maze.block(px, py, sy - px, sy - py);

    recursizeDivisionMaze(sub);
    maze.block(px, py, sy - px, sy - py) = sub;
  }
}

void
Maps::maze2D()
{
  int type = 1;
  int mx = info.sizeX / (width * info.scale);
  int my = info.sizeY / (width * info.scale);

  Eigen::MatrixXi maze(mx, my);
  maze.setZero();

  switch (type)
  {
    case 1:
      recursiveDivision(0, maze.cols() - 1, 0, maze.rows() - 1, maze);
      break;
  }

  if (addWallX)
  {
    for (int i = 0; i < mx; ++i)
    {
      maze(i, 0)      = 1;
      maze(i, my - 1) = 1;
    }
  }
  if (addWallY)
  {
    for (int i = 0; i < my; ++i)
    {
      maze(0, i)      = 1;
      maze(mx - 1, i) = 1;
    }
  }

  // std::cout << maze << std::endl;

  for (int i = 0; i < mx; ++i)
  {
    for (int j = 0; j < my; ++j)
    {
      if (maze(i, j))
      {
        for (int ii = 0; ii < width * info.scale; ++ii)
        {
          for (int jj = 0; jj < width * info.scale; ++jj)
          {
            for (int k = 0; k < info.sizeZ; ++k)
            {
              pcl::PointXYZ pt_random;
              pt_random.x =
                i * width + ii / info.scale - info.sizeX / (2.0 * info.scale);
              pt_random.y =
                j * width + jj / info.scale - info.sizeY / (2.0 * info.scale);
              pt_random.z = k / info.scale;
              info.cloud->points.push_back(pt_random);
            }
          }
        }
      }
    }
  }
  info.cloud->width    = info.cloud->points.size();
  info.cloud->height   = 1;
  info.cloud->is_dense = true;
}

Maps::BasicInfo
Maps::getInfo() const
{
  return info;
}

void
Maps::setInfo(const BasicInfo& value)
{
  info = value;
}

void
Maps::setParam(const YAML::Node& config)
{
  // perlin3D
  complexity = config["complexity"].as<double>();
  fill = config["fill"].as<double>();
  fractal = config["fractal"].as<int>();
  attenuation = config["attenuation"].as<double>();
  // randomMap
  _w_l = config["width_min"].as<double>();
  _w_h = config["width_max"].as<double>();
  _ObsNum = config["obstacle_number"].as<int>();
  // maze2D
  width = config["road_width"].as<double>();
  addWallX = config["add_wall_x"].as<int>();
  addWallY = config["add_wall_y"].as<int>();
  // tree
  tree_file = config["tree_file"].as<std::string>();
  tree_dist = config["tree_dist"].as<double>();
  // room
  room_number = config["room_number"].as<int>();
  max_windows = config["max_windows"].as<int>();
  add_ceiling = config["add_ceiling"].as<int>();
  window_size_min = config["window_size_min"].as<double>();
  window_size_max = config["window_size_max"].as<double>();
  // wall
  _wall_w_l = config["wall_width_min"].as<double>();
  _wall_w_h = config["wall_width_max"].as<double>();
  _wall_thick = config["wall_thick"].as<double>();
  _wall_num = config["wall_number"].as<int>();
  _wall_ceiling = config["wall_ceiling"].as<int>();
  // dynamic arena
  pole_num        = config["pole_number"]     ? config["pole_number"].as<int>()          : 20;
  pole_radius_min = config["pole_radius_min"] ? config["pole_radius_min"].as<double>()   : 0.15;
  pole_radius_max = config["pole_radius_max"] ? config["pole_radius_max"].as<double>()   : 0.5;
  arena_ceiling   = config["arena_ceiling"]   ? config["arena_ceiling"].as<int>()        : 0;
  arena_walls     = config["arena_walls"]     ? config["arena_walls"].as<int>()          : 1;
}


void
Maps::generate(int type)
{
  switch (type)
  {
    default:
    case 1:
      perlin3D();
      break;
    case 2:
      randomMapGenerate();
      break;
    case 3:
      std::srand(info.seed);
      maze2D();
      break;
    case 4: // generating 3d maze
      std::srand(info.seed);
      Maze3DGen();
      break;
    case 5:
      forest();
      break;
    case 6:
      room();
      break;
    case 7:
      wall();
      break;
    case 8:
      std::srand(info.seed);
      dynamicArena();
      break;
    case 9:
      std::srand(info.seed);
      corridorCrossing();
      break;
  }
}

pcl::PointXYZ
MazePoint::getPoint()
{
  return point;
}

int
MazePoint::getPoint1()
{
  return point1;
}

int
MazePoint::getPoint2()
{
  return point2;
}

double
MazePoint::getDist1()
{
  return dist1;
}

double
MazePoint::getDist2()
{
  return dist2;
}

void
MazePoint::setPoint(pcl::PointXYZ p)
{
  point = p;
}

void
MazePoint::setPoint1(int p)
{
  point1 = p;
}

void
MazePoint::setPoint2(int p)
{
  point2 = p;
}

void
MazePoint::setDist1(double set)
{
  dist1 = set;
}

void
MazePoint::setDist2(double set)
{
  dist2 = set;
}

void
Maps::Maze3DGen()
{
  // getting required info parameters from the given node
  int    numNodes = 64;
  double connectivity = 0.5;
  int    nodeRad = 4;
  int    roadRad = 3;

  // info.nh_private->param("numNodes", numNodes, 10);
  // info.nh_private->param("connectivity", connectivity, 0.5);
  // info.nh_private->param("nodeRad", nodeRad, 3);
  // info.nh_private->param("roadRad", roadRad, 2);
  // printf("received parameters : numNodes: %d connectivity: "
  //          "%f nodeRad: %d roadRad: %d",
  //          numNodes,
  //          connectivity,
  //          nodeRad,
  //          roadRad);
  // generating random points
  std::vector<pcl::PointXYZ> base;

  for (int i = 0; i < numNodes; i++)
  {
    double rx = std::rand() / RAND_MAX +
                (std::rand() % info.sizeX) / info.scale -
                info.sizeX / (2 * info.scale);
    double ry = std::rand() / RAND_MAX +
                (std::rand() % info.sizeY) / info.scale -
                info.sizeY / (2 * info.scale);
    double rz = std::rand() / RAND_MAX +
                (std::rand() % info.sizeZ) / info.scale -
                info.sizeZ / (2 * info.scale);
    // printf("point: x: %f , y: %f , z: %f", rx, ry, rz);

    pcl::PointXYZ pt_random;
    pt_random.x = rx;
    pt_random.y = ry;
    pt_random.z = rz;
    base.push_back(pt_random);
  } // generating random cores in the space

  for (int i = 0; i < info.sizeX; i++)
  {
    for (int j = 0; j < info.sizeY; j++)
    {
      for (int k = 0; k < info.sizeZ; k++)
      { // for every scaled coordinate points
        pcl::PointXYZ test;
        test.x = i / info.scale - info.sizeX / (2 * info.scale);
        test.y = j / info.scale - info.sizeY / (2 * info.scale);
        test.z = k / info.scale -
                 info.sizeZ /
                   (2 * info.scale); // marking the corresponding point location

        MazePoint mp;
        mp.setPoint(test);
        mp.setPoint2(-1);
        mp.setPoint1(-1);
        mp.setDist1(10000.0);
        mp.setDist2(100000.0); // setting super large starting values
        for (int ii = 0; ii < numNodes; ii++)
        {
          double dist =
            std::sqrt((base[ii].x - test.x) * (base[ii].x - test.x) +
                      (base[ii].y - test.y) * (base[ii].y - test.y) +
                      (base[ii].z - test.z) * (base[ii].z - test.z));
          if (dist < mp.getDist1())
          {

            mp.setDist2(mp.getDist1());
            mp.setDist1(dist);

            mp.setPoint2(mp.getPoint1());
            mp.setPoint1(ii);
          }
          else if (dist < mp.getDist2())
          {
            mp.setDist2(dist);
            mp.setPoint2(ii);
          } // finding the distances to the nearest two cores
        }
        if (std::abs(mp.getDist2() - mp.getDist1()) < 1 / info.scale)
        { // the tested location is on one of the middle planes
          if ((mp.getPoint1() + mp.getPoint2()) >
                int((1 - connectivity) * numNodes) &&
              (mp.getPoint1() + mp.getPoint2()) <
                int((1 + connectivity) * numNodes))
          { // this is a holed wall
            double judge =
              std::sqrt((base[mp.getPoint1()].x - base[mp.getPoint2()].x) *
                          (base[mp.getPoint1()].x - base[mp.getPoint2()].x) +
                        (base[mp.getPoint1()].y - base[mp.getPoint2()].y) *
                          (base[mp.getPoint1()].y - base[mp.getPoint2()].y) +
                        (base[mp.getPoint1()].z - base[mp.getPoint2()].z) *
                          (base[mp.getPoint1()].z - base[mp.getPoint2()].z));
            if (mp.getDist1() + mp.getDist2() - judge >=
                roadRad / (info.scale * 3))
            {
              info.cloud->points.push_back(mp.getPoint());
            }
          }
          else
          {
            info.cloud->points.push_back(mp.getPoint());
          }
        }
      }
    }
  }

  info.cloud->width  = info.cloud->points.size();
  info.cloud->height = 1;
  // printf("the number of points before optimization is %d", info.cloud->width);
  info.cloud->points.resize(info.cloud->width * info.cloud->height);
}

/* --------------------- My: Forest --------------------- */
void Maps::forest()
{
  double _resolution = 1 / info.scale;
  double map_width = info.sizeX / info.scale;
  double map_height = info.sizeY / info.scale;

  pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPLYFile(tree_file, *tree_cloud) == -1)
  {
    ROS_ERROR("Error: Cannot read the tree PLY file. Please check the config.yaml.");
    return;
  }

  // Poisson-distributed tree positions
  std::vector<Eigen::Vector2f> positions;
  generatePoissonPoints(map_width, map_height, tree_dist, positions);

  // generate the forest point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr forest_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  std::default_random_engine eng(info.seed);
  std::uniform_real_distribution<float> scale_dist(0.5f, 1.0f);
  std::uniform_real_distribution<float> random_angle(0.0f, 1.0f);

  for (const auto &pos : positions)
  {
    float scale_factor = scale_dist(eng);

    float roll = random_angle(eng) * 10.0f * M_PI / 180.0f;  // roll angle of 0-10 deg
    float pitch = random_angle(eng) * 10.0f * M_PI / 180.0f; // pitch angle of 0-10 deg
    float yaw = random_angle(eng) * 360.0f * M_PI / 180.0f;  // yaw angle of 0-360 deg

    Eigen::Matrix3f rotation;
    rotation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_tree(new pcl::PointCloud<pcl::PointXYZ>(*tree_cloud));
    scaleAndTranslateCloud(transformed_tree, scale_factor, pos, rotation);
    *info.cloud += *transformed_tree;
  }

  // generate the ground point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud = generateGround(info.cloud, _resolution);
  *info.cloud += *ground_cloud;

  info.cloud->width = info.cloud->points.size();
  info.cloud->height = 1;
  info.cloud->is_dense = true;
}

void Maps::generatePoissonPoints(float map_width, float map_height, float dist, std::vector<Eigen::Vector2f> &positions)
{
  float x_offset = map_width / 2.0f;
  float y_offset = map_height / 2.0f;
  
  int rows = static_cast<int>(map_width / dist);
  int cols = static_cast<int>(map_height / dist);

  std::default_random_engine eng(info.seed);
  std::uniform_real_distribution<float> offset_dist(0.0f, dist);

  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      float x = i * dist + offset_dist(eng) - x_offset;
      float y = j * dist + offset_dist(eng) - y_offset;
      positions.emplace_back(x, y);
    }
  }
}

void Maps::scaleAndTranslateCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float scale_factor, Eigen::Vector2f position, Eigen::Matrix3f &rotation)
{
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.translation() << position.x(), position.y(), 0.0f;
  transform.linear() = rotation * Eigen::Matrix3f::Identity() * scale_factor;
  pcl::transformPointCloud(*cloud, *cloud, transform);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Maps::generateGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr &forest_cloud, float grid_size, float hight)
{
  pcl::PointXYZ min_point, max_point;
  pcl::getMinMax3D(*forest_cloud, min_point, max_point);
  float x_min = min_point.x;
  float x_max = max_point.x;
  float y_min = min_point.y;
  float y_max = max_point.y;

  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  for (float x = x_min; x <= x_max; x += grid_size)
  {
    for (float y = y_min; y <= y_max; y += grid_size)
    {
      ground_cloud->emplace_back(x, y, hight);
    }
  }
  return ground_cloud;
}

/* --------------------- My: Room --------------------- */
void Maps::room()
{
  double _resolution = 1 / info.scale;
  double room_L = info.sizeX / (info.scale * (double)room_number);
  double room_W = 0.2;
  double room_H = info.sizeZ / info.scale;

  Eigen::Matrix3f rotation0 = Eigen::AngleAxisf(0, Eigen::Vector3f::UnitZ()).matrix();         // 0 deg rotation
  Eigen::Matrix3f rotation90 = Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f::UnitZ()).matrix(); // 90 deg rotation
  Eigen::Vector3f translation;

  window_eng = std::default_random_engine(info.seed);
  std::uniform_int_distribution<int> random_window(0, 100);
  dis_window_x = std::uniform_real_distribution<double>(0.1, 0.9); // window center range
  dis_window_z = std::uniform_real_distribution<double>(0.1, 0.9);
  dis_window_size = std::uniform_real_distribution<double>(window_size_min, window_size_max);

  // generate walls on a grid
  for (int i = 0; i < room_number + 1; ++i)
  {
    for (int j = 0; j < room_number + 1; ++j)
    {
      // horizontal wall (0 deg rotation)
      if (i < room_number)
      {
        pcl::PointCloud<pcl::PointXYZ>::Ptr base_wall(new pcl::PointCloud<pcl::PointXYZ>);
        int num_windows = random_window(window_eng) % max_windows + 1; // random count from 1 to max_windows
        generateWallWithWindows(base_wall, room_L, room_W, room_H, num_windows);

        translation = Eigen::Vector3f(i * room_L, j * room_L, 0);
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_wall0(new pcl::PointCloud<pcl::PointXYZ>);
        transformPointCloud(base_wall, transformed_wall0, rotation0, translation);
        *info.cloud += *transformed_wall0;
      }

      // vertical wall (90 deg rotation)
      if (j < room_number)
      {
        pcl::PointCloud<pcl::PointXYZ>::Ptr base_wall(new pcl::PointCloud<pcl::PointXYZ>);
        int num_windows = random_window(window_eng) % max_windows + 1; // random count from 1 to max_windows
        generateWallWithWindows(base_wall, room_L, room_W, room_H, num_windows);

        translation = Eigen::Vector3f(i * room_L, j * room_L, 0);
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_wall90(new pcl::PointCloud<pcl::PointXYZ>);
        transformPointCloud(base_wall, transformed_wall90, rotation90, translation);
        *info.cloud += *transformed_wall90;
      }
    }
  }
  if (add_ceiling)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud = generateGround(info.cloud, _resolution);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ceiling_cloud = generateGround(info.cloud, _resolution, room_H - _resolution);
    *info.cloud += *ground_cloud;
    *info.cloud += *ceiling_cloud;
  }
  info.cloud->width = info.cloud->points.size();
  info.cloud->height = 1;
  info.cloud->is_dense = true;
}

// Generate a base wall with windows
void Maps::generateWallWithWindows(pcl::PointCloud<pcl::PointXYZ>::Ptr wall, float L, float W, float H, int num_windows)
{
  // position and size of each window
  std::vector<std::tuple<float, float, float, float>> windows; // (x, z, width, height)

  // randomly generate windows
  for (int i = 0; i < num_windows; ++i)
  {
    float window_x = dis_window_x(window_eng) * (L - 0.5); // window center x
    float window_z = dis_window_z(window_eng) * (H - 0.5); // window center z
    float window_width = dis_window_size(window_eng);      // window width
    float window_height = dis_window_size(window_eng);     // window height

    // keep window width and height within the wall size
    window_width = std::min(window_width, L - window_x);
    window_height = std::min(window_height, H - window_z);

    // window bounds from the center
    float window_x_left = window_x - window_width / 2.0f;
    float window_z_bottom = window_z - window_height / 2.0f;

    windows.emplace_back(window_x_left, window_z_bottom, window_width, window_height);
  }

  // generate the wall point cloud, skipping window regions
  for (float x = 0; x <= L; x += 0.1f)
  {
    for (float y = 0; y <= W; y += 0.1f)
    {
      for (float z = 0; z <= H; z += 0.1f)
      {
        bool is_in_window = false;

        // check whether the current point lies in any window
        for (const auto &win : windows)
        {
          float win_x_left, win_z_bottom, win_width, win_height;
          std::tie(win_x_left, win_z_bottom, win_width, win_height) = win;

          if ((x >= win_x_left && x <= win_x_left + win_width) &&
              (z >= win_z_bottom && z <= win_z_bottom + win_height))
          {
            is_in_window = true;
            break;
          }
        }

        if (!is_in_window)
        {
          wall->points.emplace_back(x, y, z);
        }
      }
    }
  }
}

void Maps::transformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud,
                               const Eigen::Matrix3f &rotation, const Eigen::Vector3f &translation)
{
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.linear() = rotation;
  transform.translation() = translation;
  pcl::transformPointCloud(*input_cloud, *transformed_cloud, transform);
}

/* --------------------- My: Dynamic Arena --------------------- */
// Open arena: four walls + ground + optional ceiling + random static poles
// Used with swarm_dataset (moving balls) and cylinder_dataset (dynamic cylinders)
void Maps::dynamicArena()
{
  double _resolution = 1.0 / info.scale;
  double x_half = info.sizeX / (2.0 * info.scale);
  double y_half = info.sizeY / (2.0 * info.scale);
  double z_top  = info.sizeZ / info.scale;
  arena_x_half_ = static_cast<float>(x_half);
  arena_y_half_ = static_cast<float>(y_half);

  std::default_random_engine eng(info.seed);
  // leave a margin for static pole sampling so poles do not touch the walls
  double margin = std::max(pole_radius_max + 0.5, 1.0);
  std::uniform_real_distribution<double> rand_x(-x_half + margin, x_half - margin);
  std::uniform_real_distribution<double> rand_y(-y_half + margin, y_half - margin);
  std::uniform_real_distribution<double> rand_r(pole_radius_min, pole_radius_max);

  // wall thickness (one resolution layer)
  const double wall_t = _resolution;

  if (arena_walls)
  {
    // west/east walls (fixed X, sweep Y)
    for (double y = -y_half; y <= y_half; y += _resolution)
      for (double z = 0.0; z <= z_top; z += _resolution)
      {
        info.cloud->points.emplace_back(-x_half, y, z);
        info.cloud->points.emplace_back( x_half, y, z);
      }

    // south/north walls (fixed Y, sweep X)
    for (double x = -x_half; x <= x_half; x += _resolution)
      for (double z = 0.0; z <= z_top; z += _resolution)
      {
        info.cloud->points.emplace_back(x, -y_half, z);
        info.cloud->points.emplace_back(x,  y_half, z);
      }
  }

  // ground (generated from the known arena bounds, so generateGround does not break when arena_walls=0 leaves the cloud empty)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>());
    for (double x = -x_half; x <= x_half + 0.5 * _resolution; x += _resolution)
      for (double y = -y_half; y <= y_half + 0.5 * _resolution; y += _resolution)
        ground->emplace_back((float)x, (float)y, 0.0f);
    *info.cloud += *ground;
  }

  // ceiling (optional)
  if (arena_ceiling)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr ceiling(new pcl::PointCloud<pcl::PointXYZ>());
    for (double x = -x_half; x <= x_half + 0.5 * _resolution; x += _resolution)
      for (double y = -y_half; y <= y_half + 0.5 * _resolution; y += _resolution)
        ceiling->emplace_back((float)x, (float)y, (float)z_top);
    *info.cloud += *ceiling;
  }

  // random static poles: sample along the side surface only (hollow cylinders)
  static_poles_.clear();
  for (int i = 0; i < pole_num; ++i)
  {
    double cx = rand_x(eng);
    double cy = rand_y(eng);
    double r  = rand_r(eng);
    static_poles_.push_back({static_cast<float>(cx), static_cast<float>(cy),
                              static_cast<float>(r),  static_cast<float>(z_top)});
    // adapt the angular step to radius and resolution so point spacing ≤ resolution
    int n_ang = std::max(8, static_cast<int>(std::ceil(2.0 * M_PI * r / _resolution)));
    for (int ai = 0; ai < n_ang; ++ai)
    {
      double angle = 2.0 * M_PI * ai / n_ang;
      double px = cx + r * std::cos(angle);
      double py = cy + r * std::sin(angle);
      for (double z = 0.0; z <= z_top; z += _resolution)
        info.cloud->points.emplace_back(px, py, z);
    }
  }

  info.cloud->width    = info.cloud->points.size();
  info.cloud->height   = 1;
  info.cloud->is_dense = true;
}
// ─── corridor_crossing (maze_type = 9) ─────────────────────────────────────
//
// v5 design §"corridor_crossing scene": narrow corridor along X axis with two
// long side walls; a few short blocker cylinders along the corridor that the
// drone must avoid while also dodging dynamic cylinders (sim_dynamic, bounds
// confined to the corridor via arena_x_half_/arena_y_half_, see below) or CPA
// cylinders from dataset_generator.
//
// Geometry (defaults, hardcoded, not cfg-driven):
//   corridor width : 8 m
//   corridor length: full info.sizeX
//   wall height    : 3 m
//   wall thickness : 0.3 m
//   blockers       : 3 short cylinders at +X / -X / origin spaced along the
//                    corridor axis, r = 0.25 m, h = 2.5 m
void Maps::corridorCrossing()
{
  const double resolution = 1.0 / info.scale;
  const double half_x = info.sizeX / (2.0 * info.scale);
  const double half_y = info.sizeY / (2.0 * info.scale);
  const double z_top  = info.sizeZ / info.scale;

  const double corridor_half_w = 4.0;    // 8 m wide
  const double wall_thick      = 0.3;
  const double wall_height     = std::min(3.0, z_top);

  // expose corridor bounds so live-sim dynamic obstacles (test_simulator_cuda's
  // sim_dynamic) can confine their spawn/bounce range to the corridor interior
  // instead of the whole map's point-cloud bbox; also gates static-pole marker
  // publishing (arena_x_half_ > 0), same convention as dynamicArena().
  arena_x_half_ = static_cast<float>(half_x);
  arena_y_half_ = static_cast<float>(corridor_half_w);

  // ground
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground = generateGround(info.cloud, resolution);
  *info.cloud += *ground;

  // two long walls parallel to x-axis, sitting at y = ±corridor_half_w
  auto add_wall = [&](double y_center) {
    for (double x = -half_x; x <= half_x + 0.5 * resolution; x += resolution)
      for (double y = y_center - 0.5 * wall_thick; y <= y_center + 0.5 * wall_thick + 0.5 * resolution; y += resolution)
        for (double z = 0.0; z <= wall_height + 0.5 * resolution; z += resolution)
          info.cloud->points.emplace_back((float)x, (float)y, (float)z);
  };
  add_wall(+corridor_half_w);
  add_wall(-corridor_half_w);

  static_walls_.clear();
  static_walls_.push_back({0.0f, static_cast<float>(+corridor_half_w),
                            static_cast<float>(half_x), static_cast<float>(0.5 * wall_thick),
                            static_cast<float>(wall_height)});
  static_walls_.push_back({0.0f, static_cast<float>(-corridor_half_w),
                            static_cast<float>(half_x), static_cast<float>(0.5 * wall_thick),
                            static_cast<float>(wall_height)});

  // a few static blocker cylinders along the corridor axis (sparse, so the
  // dynamic cross-traffic dominates the safety challenge)
  const std::vector<double> blocker_x = {-0.4 * half_x, 0.0, +0.4 * half_x};
  const double blocker_r = 0.25;
  const double blocker_h = std::min(2.5, wall_height);

  static_poles_.clear();
  for (double cx : blocker_x)
  {
    static_poles_.push_back({static_cast<float>(cx), 0.0f,
                              static_cast<float>(blocker_r),
                              static_cast<float>(blocker_h)});
    int n_ang = std::max(8, static_cast<int>(std::ceil(2.0 * M_PI * blocker_r / resolution)));
    for (int ai = 0; ai < n_ang; ++ai)
    {
      double angle = 2.0 * M_PI * ai / n_ang;
      double px = cx + blocker_r * std::cos(angle);
      double py = 0.0 + blocker_r * std::sin(angle);
      for (double z = 0.0; z <= blocker_h + 0.5 * resolution; z += resolution)
        info.cloud->points.emplace_back(px, py, z);
    }
  }

  // (optional) ceiling — keep off by default for compatibility with downward
  // teleport sampling; enable via cfg-controlled flag if needed in future.
  (void)half_y;
  info.cloud->width    = info.cloud->points.size();
  info.cloud->height   = 1;
  info.cloud->is_dense = true;
}
