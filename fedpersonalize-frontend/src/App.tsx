import { useState, useEffect } from 'react'
import { Moon, Sun, Play, Square, Activity, Shield, Users, Zap } from 'lucide-react'
import { Button } from './components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Badge } from './components/ui/badge'
import { Progress } from './components/ui/progress'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface FLMetrics {
  round: number
  accuracy: number
  privacy_score: number
  personalization_gain: number
  clients_participated: number
  timestamp: string
}

interface FLStatus {
  status: 'idle' | 'running' | 'completed'
  current_round: number
  total_rounds: number
  clients_connected: number
  metrics: FLMetrics[]
  start_time?: string
  end_time?: string
}

function App() {
  const [darkMode, setDarkMode] = useState(false)
  const [flStatus, setFlStatus] = useState<FLStatus>({
    status: 'idle',
    current_round: 0,
    total_rounds: 0,
    clients_connected: 0,
    metrics: []
  })
  const [isLoading, setIsLoading] = useState(false)

  const API_BASE = 'http://localhost:8000'

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/fl/status`)
        const data = await response.json()
        setFlStatus(data)
      } catch (error) {
        console.error('Failed to fetch FL status:', error)
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [])

  const startFederatedLearning = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE}/fl/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_rounds: 5,
          num_clients: 3,
          local_epochs: 2,
          learning_rate: 0.01
        })
      })
      const data = await response.json()
      console.log('FL started:', data)
    } catch (error) {
      console.error('Failed to start FL:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const stopFederatedLearning = async () => {
    try {
      const response = await fetch(`${API_BASE}/fl/stop`, { method: 'POST' })
      const data = await response.json()
      console.log('FL stopped:', data)
    } catch (error) {
      console.error('Failed to stop FL:', error)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500'
      case 'completed': return 'bg-blue-500'
      default: return 'bg-gray-500'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Activity className="h-4 w-4" />
      case 'completed': return <Zap className="h-4 w-4" />
      default: return <Square className="h-4 w-4" />
    }
  }

  const progress = flStatus.total_rounds > 0 ? (flStatus.current_round / flStatus.total_rounds) * 100 : 0

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'dark bg-gray-900' : 'bg-gray-50'}`}>
      <div className="container mx-auto p-6">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
              FedPersonalize
            </h1>
            <p className="text-gray-600 dark:text-gray-300 mt-2">
              Federated Learning for IoT Personalization
            </p>
          </div>
          <Button
            variant="outline"
            size="icon"
            onClick={() => setDarkMode(!darkMode)}
            className="h-10 w-10"
          >
            {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>

        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Status</CardTitle>
              {getStatusIcon(flStatus.status)}
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-2">
                <div className={`h-2 w-2 rounded-full ${getStatusColor(flStatus.status)}`} />
                <Badge variant="secondary" className="capitalize">
                  {flStatus.status}
                </Badge>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Progress</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {flStatus.current_round}/{flStatus.total_rounds}
              </div>
              <Progress value={progress} className="mt-2" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Connected Clients</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{flStatus.clients_connected}</div>
              <p className="text-xs text-muted-foreground">IoT Devices</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Privacy Score</CardTitle>
              <Shield className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {flStatus.metrics.length > 0 
                  ? (flStatus.metrics[flStatus.metrics.length - 1].privacy_score * 100).toFixed(1)
                  : '0.0'
                }%
              </div>
              <p className="text-xs text-muted-foreground">Data Protection</p>
            </CardContent>
          </Card>
        </div>

        {/* Control Panel */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Federated Learning Control</CardTitle>
            <CardDescription>
              Start or stop the federated personalization learning process for IoT devices
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex space-x-4">
              <Button
                onClick={startFederatedLearning}
                disabled={flStatus.status === 'running' || isLoading}
                className="flex items-center space-x-2"
              >
                <Play className="h-4 w-4" />
                <span>Start Learning</span>
              </Button>
              <Button
                variant="outline"
                onClick={stopFederatedLearning}
                disabled={flStatus.status !== 'running'}
                className="flex items-center space-x-2"
              >
                <Square className="h-4 w-4" />
                <span>Stop Learning</span>
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Metrics Visualization */}
        {flStatus.metrics.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Model Accuracy</CardTitle>
                <CardDescription>Accuracy improvement over federated learning rounds</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={flStatus.metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Accuracy']} />
                    <Line 
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="#8884d8" 
                      strokeWidth={2}
                      dot={{ fill: '#8884d8' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Personalization Gain</CardTitle>
                <CardDescription>Improvement from personalized vs generic models</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={flStatus.metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis domain={[0, 0.5]} />
                    <Tooltip formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Gain']} />
                    <Line 
                      type="monotone" 
                      dataKey="personalization_gain" 
                      stroke="#82ca9d" 
                      strokeWidth={2}
                      dot={{ fill: '#82ca9d' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        )}

        {/* IoT Device Types Info */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>IoT Device Types</CardTitle>
            <CardDescription>
              Different types of smart city IoT devices participating in federated learning
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 border rounded-lg">
                <h4 className="font-semibold text-green-600 dark:text-green-400">Traffic Sensors</h4>
                <p className="text-sm text-muted-foreground mt-1">
                  Learn traffic patterns and optimize signal timing while preserving location privacy
                </p>
              </div>
              <div className="p-4 border rounded-lg">
                <h4 className="font-semibold text-blue-600 dark:text-blue-400">Environmental Sensors</h4>
                <p className="text-sm text-muted-foreground mt-1">
                  Monitor air quality and weather patterns for personalized environmental recommendations
                </p>
              </div>
              <div className="p-4 border rounded-lg">
                <h4 className="font-semibold text-purple-600 dark:text-purple-400">WiFi Access Points</h4>
                <p className="text-sm text-muted-foreground mt-1">
                  Optimize bandwidth allocation based on usage patterns without exposing user data
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
